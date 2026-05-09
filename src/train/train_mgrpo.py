"""
Training entry point for M-GRPO (Multi-query Group Relative Policy Optimization) — Stage 2.

Usage:
    deepspeed src/train/train_mgrpo.py --deepspeed ... --model_id ... --data_path ... [args]
"""

import sys
import os
import pathlib

import torch
import torch.distributed as dist

from transformers import AutoProcessor, AutoConfig, HfArgumentParser

from src.model.qwen_lvr_model import QwenWithLVR
from src.dataset.mgrpo_dataset import make_mgrpo_data_module
from src.params import DataArguments, ModelArguments, GRPOArguments
from src.train.train_utils import safe_save_model_for_hf_trainer, normalize_special_tokens
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.utils import load_reward_funcs
from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.trainer.mgrpo_trainer import QwenMGRPOTrainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def ensure_lvr_tokens_and_ids(model, processor):
    """Ensure LVR special tokens exist and sync their IDs into model.config."""
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    token_pairs = [
        ("lvr_start_id", "<|vision_start|>"),
        ("lvr_id", "<|lvr|>"),
        ("lvr_latent_end_id", "<|lvr_latent_end|>"),
        ("lvr_end_id", "<|vision_end|>"),
        ("lvr_text_start_id", "<sem>"),
        ("lvr_text_id", "<|sem|>"),
        ("lvr_text_latent_end_id", "<|sem_latent_end|>"),
        ("lvr_text_end_id", "</sem>"),
    ]

    added = 0
    vocab = tokenizer.get_vocab()
    for _, tok in token_pairs:
        if tok not in vocab:
            added += tokenizer.add_tokens(tok, special_tokens=True)

    if model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    for key, tok in token_pairs:
        setattr(model.config, key, tokenizer.convert_tokens_to_ids(tok))

    return added


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_liger_loss = False

    # OCI checkpointing (optional)
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint:
        access_key_id = os.environ.get('ACCESS_KEY_ID')
        secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('ENDPOINT_URL')
        bucket_name = os.environ.get('BUCKET_NAME')
        region_name = os.environ.get('REGION_NAME')
        model_name = model_args.model_id.split('/')[-1]
        cache_dir = os.getenv("CACHE_DIR")
        local_model_name_or_path = create_temp_dir(base_path=os.path.join(cache_dir, model_name),
                                                     prefix=training_args.run_name + '-')
        temp_folder = local_model_name_or_path
        remote_dir = os.path.join(training_args.output_dir, model_name, training_args.run_name)
        training_args.remote_output_dir = remote_dir
        training_args.output_dir = local_model_name_or_path.name
        oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name,
                                                  region_name)

    local_rank = training_args.local_rank

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # Resolve checkpoint path
    if training_args.checkpoint_name:
        model_pth = training_args.checkpoint_name
    else:
        model_pth = model_args.model_id

    # Normalize special tokens so LVR tokens can be generated
    tokens_to_normalize = {
        "<|vision_start|>", "<|vision_end|>", "<|lvr|>", "<|lvr_latent_end|>",
        "<sem>", "</sem>", "<|sem|>",
    }
    normalize_special_tokens(model_pth, tokens_to_normalize)

    # Load config
    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    if model_args.latent_end_token:
        config.latent_end_token = True
    # Keep stage-1 checkpoint behavior by default; only override when explicitly enabled.
    if model_args.lvr_text_head:
        config.lvr_text_head = True

    # Monkey-patch forward
    if "Qwen2.5" in model_args.model_id:
        # Use the comprehensive forward from monkey_patch_forward_lvr.py, not the RL-only version
        # This ensures latent decoding during generation works correctly (like in inference)
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            inference_mode=False,  # Not inference mode (training mode)
            rl=True,               # Enable RL-specific logic
            lvr_head=getattr(config, "lvr_head", False),
            latent_end_token=getattr(config, "latent_end_token", False),
        )

        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        if model_args.latent_end_token:
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        replace_qwen_2_5_vl_patch_emb()
    else:
        raise ValueError("Only Qwen2.5 VL models are supported.")

    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    # Freeze modules that are NOT used in the RL (GRPO) training forward.
    # Without this, ZeRO-2 registers gradient hooks for these params but
    # their hooks never fire (no gradient flows), causing an allreduce
    # deadlock (SeqNum 732, NumelIn=3584).
    _rl_unused_keywords = ('lvr_head', 'text_embedding_projector', 'lvr_latent_end_emb')
    _frozen = []
    for name, param in model.named_parameters():
        if any(k in name for k in _rl_unused_keywords):
            param.requires_grad = False
            _frozen.append(name)
    if _frozen:
        rank0_print(f"[M-GRPO] Froze {len(_frozen)} RL-unused params to prevent ZeRO-2 deadlock: {_frozen}")

    # NaN sanitizer hook
    def output_nan_sanitizer_hook(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        return output

    model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)

    processor = AutoProcessor.from_pretrained(model_pth)
    added_tokens = ensure_lvr_tokens_and_ids(model, processor)
    rank0_print(f"[M-GRPO] LVR token check complete. Newly added tokens: {added_tokens}")

    dataset_module = make_mgrpo_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    reward_funcs = load_reward_funcs("src.train.mgrpo_reward_funcs")
    rank0_print(f"[M-GRPO] Loaded {len(reward_funcs)} reward functions")

    trainer = QwenMGRPOTrainer(
        model=model,
        ref_model_pth=model_pth,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        reward_funcs=reward_funcs,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
