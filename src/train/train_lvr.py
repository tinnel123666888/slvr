import sys
import os
import json

import os
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_supervised_data_module_lvr, make_packed_supervised_data_module_lvr
from src.params import DataArguments, ModelArguments, TrainingArguments

from src.train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Debug: Print model args on each rank
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size}: model_id={model_args.model_id}, lvr_head={model_args.lvr_head}, lvr_text_head={model_args.lvr_text_head}, latent_end_token={model_args.latent_end_token}, lvr_head_type={model_args.lvr_head_type}, max_lvr_tokens={model_args.max_lvr_tokens}")
        # Check consistency
        all_lvr_head = [None] * world_size
        dist.all_gather_object(all_lvr_head, model_args.lvr_head)
        if not all(x == model_args.lvr_head for x in all_lvr_head):
            print(f"Rank {rank}: lvr_head mismatch: {all_lvr_head}")
            raise ValueError("lvr_head not consistent across ranks")
        # Similarly for others
        all_lvr_text_head = [None] * world_size
        dist.all_gather_object(all_lvr_text_head, model_args.lvr_text_head)
        if not all(x == model_args.lvr_text_head for x in all_lvr_text_head):
            print(f"Rank {rank}: lvr_text_head mismatch: {all_lvr_text_head}")
            raise ValueError("lvr_text_head not consistent across ranks")
        dist.barrier()
    else:
        print(f"Rank {local_rank}: model_id={model_args.model_id}, lvr_head={model_args.lvr_head}, lvr_text_head={model_args.lvr_text_head}, latent_end_token={model_args.latent_end_token}, lvr_head_type={model_args.lvr_head_type}, max_lvr_tokens={model_args.max_lvr_tokens}")

    '''
        set up oci checkpointing;
        set online_checkpoint to False if you dont need
    '''
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint:
        # oci keys
        access_key_id = os.environ.get('ACCESS_KEY_ID')
        secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('ENDPOINT_URL')
        bucket_name = os.environ.get('BUCKET_NAME')
        region_name = os.environ.get('REGION_NAME')

        model_name = model_args.model_id.split('/')[-1]     # "Qwen2.5-VL-7B-Instruct"
        # local cache dir and tempFile class
        cache_dir = os.getenv("CACHE_DIR")  #cache dir = "/dockerx/Local/users/bangzheng"
        # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
        local_model_name_or_path = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=training_args.run_name + '-')     
        temp_folder = local_model_name_or_path

        # remote dir
        remote_dir = training_args.output_dir  # output_dir is remote now; "/checkpoints"
        remote_dir = os.path.join(remote_dir,model_name,training_args.run_name)    # "/checkpoints/Qwen2.5-VL-7B-Instruct/run_name"
        training_args.remote_output_dir = remote_dir
        training_args.output_dir = local_model_name_or_path.name    # output_dir should always be local

        # oci handler
        oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name, region_name)
    

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        if training_args.online_checkpoint:
            # CHKPT_NAME="checkpoints_lvrHead_featureAlign/Qwen2.5-VL-7B-Instruct/BS256-LAMBDA1-LVR_HEAD_LR1e-5-MAXTOKEN{7680}/checkpoint-1578/"
            local_pth_to_download_chkpt = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=f"warmed_{model_args.lvr_head_type}" + '-')
            oci_handler.load_checkpoint(training_args.checkpoint_name, local_pth_to_download_chkpt,inference_mode=True)
            
            model_pth = local_pth_to_download_chkpt.name
        else:
            model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        print(f"Rank {rank}: Loaded config from {model_pth}: lvr_head={getattr(config, 'lvr_head', None)}, lvr_text_head={getattr(config, 'lvr_text_head', None)}, latent_end_token={getattr(config, 'latent_end_token', None)}, lvr_head_type={getattr(config, 'lvr_head_type', None)}")
        dist.barrier()
    else:
        print(f"Rank {local_rank}: Loaded config from {model_pth}: lvr_head={getattr(config, 'lvr_head', None)}, lvr_text_head={getattr(config, 'lvr_text_head', None)}, latent_end_token={getattr(config, 'latent_end_token', None)}, lvr_head_type={getattr(config, 'lvr_head_type', None)}")
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head
    config.lvr_text_head = model_args.lvr_text_head
    config.lvr_head_type = model_args.lvr_head_type
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        print(f"Rank {rank}: Updated config: lvr_head={config.lvr_head}, lvr_text_head={config.lvr_text_head}, latent_end_token={config.latent_end_token}, lvr_head_type={config.lvr_head_type}")
        dist.barrier()
    else:
        print(f"Rank {local_rank}: Updated config: lvr_head={config.lvr_head}, lvr_text_head={config.lvr_text_head}, latent_end_token={config.latent_end_token}, lvr_head_type={config.lvr_head_type}")
    
    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function
        replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=model_args.coconut,
                                                        lvr_head=model_args.lvr_head,
                                                        mode_switch_loss=training_args.mode_switch_loss,
                                                        latent_end_token=model_args.latent_end_token)
        
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        # init lvr_head
        if model_args.lvr_head:
            model._init_lvr_head(lvr_head_type =  model_args.lvr_head_type)
        
        # init latent_end_token
        if model_args.latent_end_token:
            model._init_lvr_latent_end_emb()
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        
        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")
    
    # Debug: Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}: Model total parameters: {total_params}")
        # Check consistency
        all_params = [None] * world_size
        dist.all_gather_object(all_params, total_params)
        if not all(x == total_params for x in all_params):
            print(f"Rank {rank}: Parameter count mismatch: {all_params}")
            raise ValueError("Model parameter count not consistent across ranks")
        dist.barrier()
    else:
        print(f"Rank {local_rank}: Model total parameters: {total_params}")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    ''' NaN sanitizer: Hook the patch-emb with torch.nan_to_num() '''
    # def output_nan_sanitizer_hook(module, input, output):
    #     if isinstance(output, torch.Tensor) and torch.isnan(output).any():
    #         print(f"[Sanitizer] {module.__class__.__name__}: NaN or Inf detected.")
    #         print(f"  Output stats - min: {output.min().item()}, max: {output.max().item()}, mean: {output.mean().item()}")
    #         return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
    #     return output
    # model.model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)

    processor.tokenizer.add_tokens("<|vision_start|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_latent_end|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|vision_end|>",special_tokens=True)

    lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    #lvr text
    processor.tokenizer.add_tokens("<sem>", special_tokens=True)
    processor.tokenizer.add_tokens("<|sem|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|sem_latent_end|>",special_tokens=True)
    processor.tokenizer.add_tokens("</sem>", special_tokens=True)

    # 获取对应的token ID
    lvr_text_start_id = processor.tokenizer.convert_tokens_to_ids("<sem>")
    lvr_text_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|sem_latent_end|>")
    lvr_text_id = processor.tokenizer.convert_tokens_to_ids("<|sem|>")
    lvr_text_end_id = processor.tokenizer.convert_tokens_to_ids("</sem>")

    # 将它们添加到模型配置中
    model.config.lvr_text_start_id = lvr_text_start_id
    model.config.lvr_text_id = lvr_text_id
    model.config.lvr_text_latent_end_id = lvr_text_latent_end_id
    model.config.lvr_text_end_id = lvr_text_end_id

    model.config.lvr_id = lvr_id
    model.config.lvr_latent_end_id = lvr_latent_end_id
    model.config.lvr_start_id = lvr_start_id
    model.config.lvr_end_id = lvr_end_id


    # there are some dummy tokens in newer hf version
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure lvr loss type
    model.config.loss_lvr_fct = training_args.loss_lvr_fct


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LVR
    '''
    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        if model_args.max_lvr_tokens is not None:
            data_module, total_data_len = make_packed_supervised_data_module_lvr_fixedToken(model_id=model_args.model_id,
                                                                                            processor=processor,
                                                                                            max_lvr_tokens=model_args.max_lvr_tokens,
                                                                                            data_args=data_args,
                                                                                            training_args=training_args,
                                                                                            latent_end_token=model_args.latent_end_token)
        else:
            data_module, total_data_len = make_packed_supervised_data_module_lvr(model_id=model_args.model_id,
                                                                                processor=processor,
                                                                                data_args=data_args,
                                                                                training_args=training_args,
                                                                                latent_end_token=model_args.latent_end_token)
        if training_args.max_steps is None or training_args.max_steps <= 0:
            training_args.max_steps = total_data_len // (training_args.gradient_accumulation_steps 
                                                            * training_args.world_size
                                                            * training_args.per_device_train_batch_size)
            training_args.max_steps = max(training_args.max_steps, 1)
            rank0_print(f"[AutoSteps] computed max_steps={training_args.max_steps} from total_data_len={total_data_len}")
        # Very crucial or the packed data will get incorrectly sliced by the dataloader
        replace_train_dataloader()
    else:
        # Non-packed pipeline expects the data file itself to be a list of training samples.
        # If user passes a meta file (e.g. [{"ds_name", "data_path", "image_folder", ...}]),
        # fail early with a clear message instead of KeyError on missing fields like `bboxes`.
        try:
            with open(data_args.data_path, "r", encoding="utf-8") as f:
                maybe_meta = json.load(f)
            if (
                isinstance(maybe_meta, list)
                and len(maybe_meta) > 0
                and isinstance(maybe_meta[0], dict)
                and "data_path" in maybe_meta[0]
                and "ds_name" in maybe_meta[0]
            ):
                raise ValueError(
                    f"data_path={data_args.data_path} appears to be a meta dataset config. "
                    "Please enable data packing (--enable_data_packing True), or switch --data_path "
                    "to a concrete sample list JSON with fields like 'conversations', 'image', 'bboxes'."
                )
        except ValueError:
            raise
        except Exception:
            # Keep backward compatibility if probing fails for any non-critical reason.
            pass

        data_module = make_supervised_data_module_lvr(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              latent_end_token=model_args.latent_end_token)
    
    # tempFolder = temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module
    )

    trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
