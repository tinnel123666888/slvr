import torch
import os
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.utils import is_torchdynamo_compiling,TransformersKwargs
from transformers.processing_utils import Unpack
from src.constants import IGNORE_INDEX
import torch.distributed as dist

import torch.nn.functional as F


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def replace_qwen2_5_with_mixed_modality_forward_slvr(inference_mode=False,
                                                    coconut=True,
                                                    slvr_head=True,
                                                    mode_switch_loss=False,
                                                    latent_end_token=False,
                                                    rl = False):
    verbose = os.getenv("MGRPO_FORWARD_PATCH_VERBOSE", "0") == "1"

    if inference_mode:
        if slvr_head:
            if verbose:
                print("Inference mode with Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_with_head_inference
        else:
            if verbose:
                print("Inference mode without Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_inference
    elif rl:
        if verbose:
            print("Activated stage 2 training!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_rl
    else:
        if latent_end_token and slvr_head:
            if verbose:
                print("Activated latent end token mode with SLVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_with_head_with_latentEndToken
        elif latent_end_token and not slvr_head:
            if verbose:
                print("Activated latent end token mode without SLVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_with_latentEndToken
        elif mode_switch_loss:
            if verbose:
                print("Activated BCE mode swtich loss!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_with_head_with_modeSwitchLoss
        elif slvr_head:
            if verbose:
                print("Activated naive SLVR with head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr_with_head
        else:
            if verbose:
                print("Activated naive SLVR without head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_slvr


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_slvr: Optional[torch.FloatTensor] = None
    loss_slvr_text: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mode_switch: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # next_pos_slvr:Optional[bool] = False


def  set_slvr_loss_fct(loss_slvr_fct: str):
    """
        Set the loss function for SLVR.
        Args:
            loss_slvr_fct (str): The type of loss function to use for SLVR.
        Returns:
            A loss function object.
    """
    if loss_slvr_fct == 'mse':
        return MSELoss()
    elif loss_slvr_fct == 'mae':
        return L1Loss()
    elif loss_slvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported slvr_loss: {loss_slvr_fct}")

'''
    Coconut mode
    No SLVR Head
'''
def qwen2_5_mixed_modality_forward_slvr(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    text_embedding: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (
        last_position_hidden_state is not None
        and slvr_mode_switch is not None
        and (not isinstance(slvr_mode_switch, torch.Tensor) or slvr_mode_switch.any())
    ):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    no_slvr_mode = (
        slvr_mode_switch is None
        or (isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch.any())
        or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)
    )
    if no_slvr_mode and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        # print(image_embeds)
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens is not None:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  
            
            if isinstance(slvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_slvr_token_indices = []
                # print(slvr_tokens)
                for b, slvr_ids in enumerate(slvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_slvr_token_indices.append(slvr_ids + offset)
                global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                if batch_indices.numel() > 0:
                    # print('-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                    inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
                else:
                    # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
                    # 可选：打印警告，但注意在分布式训练中这会产生很多日志
                    # 或者什么都不做，因为如果整个批次都没有SLVR tokens，那么就不需要替换
                    pass
                # inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
            else:
                '''re-encode target area'''
                # Now slvr_tokens is pixel_values of the cropped targets
                selected_slvr_embeds = self.model.get_image_features(slvr_tokens, slvr_tokens_thw)
                selected_slvr_embeds = torch.cat(selected_slvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
        # 处理完SLVR tokens后，添加text embedding处理逻辑
        if text_embedding is not None and hasattr(self.config, 'slvr_text_id'):
            projected_text_embeddings = None 
            # 1. 识别<|sem|> token的位置
            slvr_text_mask = input_ids == self.config.slvr_text_id
            batch_indices_text, seq_positions_text = torch.nonzero(slvr_text_mask, as_tuple=True)
            
            # Ensure text_embedding_projector is always used to avoid None gradients
            dummy = self.text_embedding_projector(torch.zeros(1, 4096, device=self.device, dtype=self.dtype)).sum() * 1e-20
            
            if slvr_text_mask.any():
                batch_size = input_ids.size(0)
                
                # 调试信息
                # print(f"text_embedding shape: {text_embedding.shape}")
                # print(f"batch_size: {batch_size}")
                
                # 根据text_embedding的形状进行处理
                if text_embedding.dim() == 1:
                    # 如果是1D向量 [16384]，需要reshape为 [4, 4096]
                    total_elements = text_embedding.numel()
                    if total_elements == batch_size * 4096:
                        # 确认是4个4096维向量的拼接
                        text_embedding = text_embedding.view(batch_size, 4096)
                    else:
                        raise ValueError(f"text_embedding has {total_elements} elements, "
                                    f"expected {batch_size} * 4096 = {batch_size * 4096}")
                
                elif text_embedding.dim() == 2:
                    # 如果是2D张量
                    if text_embedding.size(0) == 1 and text_embedding.size(1) == batch_size * 4096:
                        # 形状为 [1, 16384]，需要reshape为 [4, 4096]
                        text_embedding = text_embedding.view(batch_size, 4096)
                    elif text_embedding.size(0) == batch_size and text_embedding.size(1) == 4096:
                        # 形状已经是 [4, 4096]，无需修改
                        pass
                    else:
                        raise ValueError(f"Unsupported 2D text_embedding shape: {text_embedding.shape}")
                
                else:
                    raise ValueError(f"text_embedding has {text_embedding.dim()} dimensions, expected 1 or 2")
                
                # 确保text_embedding的形状是 [batch_size, 4096]
                assert text_embedding.shape == (batch_size, 4096), \
                    f"Expected shape ({batch_size}, 4096), got {text_embedding.shape}"
                
                # 投影到模型的hidden_size
                projected_text_embeddings = self.text_embedding_projector(text_embedding)
                
                # 替换embeddings - 在 <|sem|> 位置（对标 <|slvr|>，嵌入在 token 位置）
                inputs_embeds[batch_indices_text, seq_positions_text] = projected_text_embeddings[batch_indices_text].to(inputs_embeds.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
    slvr_text_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
    loss = None
    loss_ce = None
    loss_slvr = None
    loss_slvr_text = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_text_id, IGNORE_INDEX)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)
        if text_embedding is not None and hasattr(self.config, 'slvr_text_id'):
            loss_ce += dummy

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to vision_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)
        if text_embedding is not None and hasattr(self.config, 'slvr_text_id') and projected_text_embeddings is not None:
            # 重新识别<|sem|>的位置
            slvr_text_mask = input_ids == self.config.slvr_text_id
            batch_indices_text, seq_positions_text = torch.nonzero(slvr_text_mask, as_tuple=True)
            
            if slvr_text_mask.any():
                # 获取<sem>位置（<|sem|>的前一个位置）
                seq_positions_text_start = seq_positions_text - 1
                
                # 获取这些位置的hidden states
                selected_hidden_states_text = hidden_states[batch_indices_text, seq_positions_text_start].to(torch.float32)
                
                # 确保projected_text_embeddings的形状正确
                # 需要根据batch_indices_text选择对应的嵌入
                target_text_embeddings = projected_text_embeddings[batch_indices_text].to(torch.float32)
                
                loss_slvr_text = slvr_text_loss_fct(selected_hidden_states_text, target_text_embeddings)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        loss_slvr_text=loss_slvr_text,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    No SLVR Head
'''
def qwen2_5_mixed_modality_forward_slvr_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,
    slvr_tokens_thw: Optional[List[torch.Tensor]] = None,
    slvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    text_embedding: Optional[torch.FloatTensor] = None,  # Added for text embedding support
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(self.model.visual.device)
    
    # Unified handling for SLVR mode replacement (removed duplicate block)
    if slvr_mode_switch is not None and last_position_hidden_state is not None:
        # Only replace for instances in SLVR mode
        if slvr_mode_switch.any():
            last_position_hidden_state = last_position_hidden_state.to(self.model.visual.device)
            inputs_embeds[slvr_mode_switch, -1, :] = last_position_hidden_state[slvr_mode_switch]

    # Dummy image handling for text-only batches (prevents deepspeed errors)
    if (pixel_values is None and pixel_values_videos is None) and (
        slvr_mode_switch is None or 
        (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch))
    ):
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds.to(self.model.visual.device) + image_embeds.mean().to(self.model.visual.device) * 0  # No-op that triggers computation
            
    # Process real images if available
    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
    
        # Validate image token count matches embeddings
        if input_ids is None:
            image_mask = (inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )).all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = image_mask.sum()
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and tokens mismatch: tokens={n_image_tokens}, features={n_image_features}"
            )
        
        # Inject image embeddings into input embeddings
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze.to(inputs_embeds.device, image_mask.dtype), image_embeds)

        # Handle SLVR tokens during inference (same logic as training)
        if slvr_tokens is not None:
            total_tokens = torch.sum(image_mask, dim=1)
            slvr_mask = input_ids == self.config.slvr_id
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)

            if isinstance(slvr_tokens, list):
                # Extract tokens from original image
                image_token_offsets = torch.cumsum(F.pad(total_tokens, (1, 0)), dim=0)[:-1]
                global_slvr_token_indices = [
                    slvr_ids + image_token_offsets[b] 
                    for b, slvr_ids in enumerate(slvr_tokens)
                ]
                global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)
                selected_slvr_embeds = image_embeds[global_slvr_token_indices]
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
            else:
                # Re-encode cropped regions
                selected_slvr_embeds = self.model.get_image_features(slvr_tokens, slvr_tokens_thw)
                selected_slvr_embeds = torch.cat(selected_slvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds

    # Handle text embeddings for SLVR text tokens
    if text_embedding is not None and hasattr(self.config, 'slvr_text_id'):
        slvr_text_mask = input_ids == self.config.slvr_text_id
        batch_indices_text, seq_positions_text = torch.nonzero(slvr_text_mask, as_tuple=True)
        
        if slvr_text_mask.any():
            batch_size = input_ids.size(0)
            
            # Handle different text_embedding shapes
            if text_embedding.dim() == 1:
                total_elements = text_embedding.numel()
                if total_elements == batch_size * 4096:
                    text_embedding = text_embedding.view(batch_size, 4096)
                else:
                    raise ValueError(f"Unexpected text_embedding size: {total_elements}")
            elif text_embedding.dim() == 2:
                if text_embedding.size(0) == 1 and text_embedding.size(1) == batch_size * 4096:
                    text_embedding = text_embedding.view(batch_size, 4096)
                elif not (text_embedding.size(0) == batch_size and text_embedding.size(1) == 4096):
                    raise ValueError(f"Invalid text_embedding shape: {text_embedding.shape}")
            
            # Project and inject text embeddings
            projected_text_embeddings = self.text_embedding_projector(text_embedding)
            inputs_embeds[batch_indices_text, seq_positions_text] = (
                projected_text_embeddings[batch_indices_text].to(inputs_embeds.device)
            )

    # Prepare attention mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    # Position ID handling
    if position_ids is None:
        prefill_compiled = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1) or
            (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0) or
            (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        
        if prefill_compiled or prefill_noncompiled or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.repeat(batch_size, 1)
            if cache_position is not None:
                delta = cache_position[0] + self.model.rope_deltas
            else:
                delta = torch.zeros(batch_size, device=inputs_embeds.device)
            position_ids += delta.unsqueeze(1)

    # Language model forward pass
    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    hidden_states = outputs[0]
    last_position_hidden_state = hidden_states[:, -1, :]  # Save last hidden state
    logits = self.lm_head(hidden_states)

    # Loss computation (for validation mode)
    loss_ce = loss_slvr = loss_slvr_text = None
    if labels is not None:
        # Standard CE loss (ignore SLVR tokens)
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(
            (shift_labels == self.config.slvr_id) | 
            (shift_labels == self.config.slvr_text_id),
            IGNORE_INDEX
        )
        
        loss_fct = CrossEntropyLoss()
        loss_ce = loss_fct(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1).to(shift_logits.device)
        )

        # SLVR embedding loss (if applicable)
        if slvr_tokens is not None and 'batch_indices' in locals() and batch_indices.numel() > 0:
            seq_positions_start = seq_positions - 1
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].float()
            selected_slvr_embeds = selected_slvr_embeds.float()
            slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
            loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)

        # SLVR text embedding loss (if applicable)
        if text_embedding is not None and hasattr(self.config, 'slvr_text_id'):
            slvr_text_mask = input_ids == self.config.slvr_text_id
            if slvr_text_mask.any():
                batch_indices_text, seq_positions_text = torch.nonzero(slvr_text_mask, as_tuple=True)
                seq_positions_text_start = seq_positions_text - 1
                selected_hidden_states_text = hidden_states[batch_indices_text, seq_positions_text_start].float()
                target_text_embeddings = projected_text_embeddings[batch_indices_text].float()
                slvr_text_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
                loss_slvr_text = slvr_text_loss_fct(selected_hidden_states_text, target_text_embeddings)

    # Prepare output
    if not return_dict:
        output = (logits,) + outputs[1:]
        losses = (loss_ce, loss_slvr, loss_slvr_text)
        non_none_losses = tuple(l for l in losses if l is not None)
        return (non_none_losses + output) if any(l is not None for l in losses) else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        loss_slvr_text=loss_slvr_text,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state
    )



'''
    Coconut mode;
    SLVR head;
    Note that this forward function is used for inferencing all the SLVR models with a SLVR head
'''
def qwen2_5_mixed_modality_forward_slvr_with_head(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (
        last_position_hidden_state is not None
        and slvr_mode_switch is not None
        and (not isinstance(slvr_mode_switch, torch.Tensor) or torch.any(slvr_mode_switch))
    ):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens is not None:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  
            if isinstance(slvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_slvr_token_indices = []
                for b, slvr_ids in enumerate(slvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_slvr_token_indices.append(slvr_ids + offset)
                global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
            
            else:
                '''re-encode target area'''
                # Now slvr_tokens is pixel_values of the cropped targets
                selected_slvr_embeds = self.model.get_image_features(slvr_tokens, slvr_tokens_thw)
                selected_slvr_embeds = torch.cat(selected_slvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply slvr_head in training mode'''
    if slvr_tokens is not None and slvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <slvr> token positions, starting <vision_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to vision_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.slvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply slvr_head in _inference mode'''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        outputs.last_hidden_state[slvr_mode_switch,:,:] = self.slvr_head(outputs.last_hidden_state[slvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to vision_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )





def qwen2_5_mixed_modality_forward_slvr_with_head_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens is not None:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  
            if isinstance(slvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_slvr_token_indices = []
                for b, slvr_ids in enumerate(slvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_slvr_token_indices.append(slvr_ids + offset)
                global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
            
            else:
                '''re-encode target area'''
                # Now slvr_tokens is pixel_values of the cropped targets
                selected_slvr_embeds = self.model.get_image_features(slvr_tokens, slvr_tokens_thw)
                selected_slvr_embeds = torch.cat(selected_slvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    '''apply slvr_head in training mode'''
    if slvr_tokens is not None and slvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <slvr> token positions, starting <vision_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to vision_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.slvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            # selected_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start]
            # slvr_head_output = self.slvr_head(selected_hidden_states)
            # outputs.last_hidden_state[batch_indices, seq_positions_start] = slvr_head_output

    '''apply slvr_head in _inference mode'''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        outputs.last_hidden_state[slvr_mode_switch,:,:] = self.slvr_head(outputs.last_hidden_state[slvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to vision_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    SLVR head
'''
def qwen2_5_mixed_modality_forward_slvr_with_head_with_modeSwitchLoss(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_slvr_token_indices = []

            for b, slvr_ids in enumerate(slvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_slvr_token_indices.append(slvr_ids + offset)
            global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply slvr_head in training mode'''
    if slvr_tokens and slvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <slvr> token positions, starting <vision_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to vision_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.slvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply slvr_head in _inference mode'''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        outputs.last_hidden_state[slvr_mode_switch,:,:] = self.slvr_head(outputs.last_hidden_state[slvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to vision_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)

        # mode switch loss

        slvr_or_slvrstart_mask = (input_ids == self.config.slvr_start_id) | (input_ids == self.config.slvr_id)

        # Find the next tokens of each position
        shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=1)
        # the slvr token that is right before vision_end token
        is_last_slvr = slvr_or_slvrstart_mask & (shifted_input_ids == self.config.slvr_end_id)
        # 1 if it's the last <slvr> before <vision_end>, else 0
        targets = is_last_slvr.float()  # [batch_size, seq_len]

        slvr_end_logits = logits[..., self.config.slvr_end_id]  # [batch_size, seq_len]

        # Apply mask to focus only on <vision_start>,<slvr> token positions
        masked_logits = slvr_end_logits[slvr_or_slvrstart_mask]  # [num_slvr_tokens]
        masked_targets = targets[slvr_or_slvrstart_mask]        # [num_slvr_tokens]

        loss_mode_switch = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    SLVR Head
    Padded <SLVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_slvr_with_head_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_slvr_token_indices = []

            for b, slvr_ids in enumerate(slvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_slvr_token_indices.append(slvr_ids + offset)
            global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds

            '''Apply slvr_latent_end_token'''
            slvr_latent_end_mask = (input_ids == self.config.slvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(slvr_latent_end_mask, as_tuple=True)
            if slvr_latent_end_mask.any():
                inputs_embeds[slvr_latent_end_mask] = self.slvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply slvr_head in training mode'''
    if slvr_tokens and slvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <slvr> token positions, starting <vision_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to vision_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.slvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            '''In this mode, <|slvr_latent_end|> is also a latent token'''
            seq_positions_start_latentend = seq_positions_latentend - 1
            outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend] = self.slvr_head(outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend])


    '''apply slvr_head in _inference mode'''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        outputs.last_hidden_state[slvr_mode_switch,:,:] = self.slvr_head(outputs.last_hidden_state[slvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
    mode_switch_loss_fct = set_slvr_loss_fct(self.config.loss_mode_switch_fct)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.slvr_id)|(shift_labels == self.config.slvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <slvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        selected_slvr_embeds_latentend = self.slvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_slvr_embeds_latentend = selected_slvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_slvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )



'''
    Coconut mode
    SLVR Head
    Padded <SLVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_slvr_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the slvr img tokens be
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (slvr_mode_switch is not None) and ((not isinstance(slvr_mode_switch, torch.Tensor)) or torch.any(slvr_mode_switch)):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the slvr token positions with selected img tokrnd
        if slvr_tokens:
            '''
                Filling the slvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # slvr mask for slvr token locations in the batch, [bs, seq_length]
            # in each instance, slvr tokens are True, others are False
            slvr_mask = input_ids == self.config.slvr_id  
            # Total length = number of <slvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_slvr_token_indices = []

            for b, slvr_ids in enumerate(slvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_slvr_token_indices.append(slvr_ids + offset)
            global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds

            '''Apply slvr_latent_end_token'''
            slvr_latent_end_mask = (input_ids == self.config.slvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(slvr_latent_end_mask, as_tuple=True)
            if slvr_latent_end_mask.any():
                inputs_embeds[slvr_latent_end_mask] = self.slvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)
    mode_switch_loss_fct = set_slvr_loss_fct(self.config.loss_mode_switch_fct)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.slvr_id)|(shift_labels == self.config.slvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # slvr loss
        # Get last hidden states for <slvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <slvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_slvr_embeds = selected_slvr_embeds.to(torch.float32)
        selected_slvr_embeds_latentend = self.slvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_slvr_embeds_latentend = selected_slvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute SLVR loss between predicted and inserted slvr embeddings
        loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_slvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


"""
    Forward function for stage 2 RL
    Kinda messy since in this stage, the transofmers will be 4.51.3 < 4.54 in stage I
    Will fix this inconsistency in final release
"""
def qwen2_5_mixed_modality_forward_slvr_rl(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    slvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in slvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    slvr_mask: Optional[torch.FloatTensor] = None,   # This is for RL loss computation
    slvr_states: Optional[torch.FloatTensor] = None, # This is for RL loss computation
    prompt_length: Optional[int] = None, # This is for RL loss computation
    text_embedding: Optional[torch.FloatTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''In this mode, no slvr_tokens'''
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if (
        last_position_hidden_state is not None
        and slvr_mode_switch is not None
        and (not isinstance(slvr_mode_switch, torch.Tensor) or torch.any(slvr_mode_switch))
    ):
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[slvr_mode_switch,-1,:] = last_position_hidden_state[slvr_mode_switch]

    # Teacher-forcing replay path used by GRPO loss: patch sampled SLVR states
    if slvr_states is not None and slvr_mask is not None and prompt_length is not None:
        comp_embeds = inputs_embeds[:, prompt_length:, :]
        comp_embeds = torch.where(
            slvr_mask.unsqueeze(-1),
            slvr_states,
            comp_embeds,
        )
        inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length, :], comp_embeds], dim=1)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if ((slvr_mode_switch is None) or (isinstance(slvr_mode_switch, torch.Tensor) and not torch.any(slvr_mode_switch)) or (not isinstance(slvr_mode_switch, torch.Tensor) and not slvr_mode_switch)) and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # check if there is slvr_head
    if self.config.slvr_head:
        '''apply slvr_head in _inference mode'''
        if slvr_mode_switch is not None:
            outputs.last_hidden_state[slvr_mode_switch,:,:] = self.slvr_head(outputs.last_hidden_state[slvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_slvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <slvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # No slvr loss in this mode
        loss_slvr = None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_slvr=loss_slvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )







'''Liger kernel'''
# def qwen2_5_mixed_modality_forward_slvr_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
#     slvr_tokens: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
#             '''
#                 Filling the slvr tokens with image embeddings.
#                 Applicable when each image input has multiple bboxes
#             '''
#             total_tokens = torch.sum(mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
#             batch_size = input_ids.size(0) 
#             # slvr mask for slvr token locations in the batch, [bs, seq_length]
#             # in each instance, slvr tokens are True, others are False
#             slvr_mask = input_ids == self.config.slvr_id  
#             # Total length = number of <slvr> tokens in the batch
#             # seq_positions: flattend LOCAL positions of slvr tokens in the inputs_ids
#             batch_indices, seq_positions = torch.nonzero(slvr_mask, as_tuple=True)  

#            #  GLOBAL starting index in `image_embeds` of each image in the batch
#             image_token_offsets = torch.cumsum(
#                 F.pad(total_tokens, (1, 0)), dim=0
#             )[:-1]  # shape [B], offset into image_embeds for each batch element

#             global_slvr_token_indices = []

#             for b, slvr_ids in enumerate(slvr_tokens):
#                 # Convert local to global index
#                 offset = image_token_offsets[b].item()
#                 global_slvr_token_indices.append(slvr_ids + offset)
#             global_slvr_token_indices = torch.cat(global_slvr_token_indices, dim=0)  # [L_total]

#             # Step 3: Gather the selected visual embeddings
#             selected_slvr_embeds = image_embeds[global_slvr_token_indices]  # [L_total, H]

#             # Step 4: Replace in input_embeds at the right batch and position
#             # Prepare indexing
#             # replaced_embeds = inputs_embeds.clone()
#             inputs_embeds[batch_indices, seq_positions] = selected_slvr_embeds



#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
#             video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]

#     slvr_loss_fct = set_slvr_loss_fct(self.config.loss_slvr_fct)


#     loss = None
#     loss_ce = None
#     loss_slvr = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)
#         # Don't want CE loss for <slvr> token
#         shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss_ce = lce(self.lm_head.weight, shift_hidden_states, shift_labels)

        
#         # slvr loss
#         # Get last hidden states for <slvr> token positions
#         seq_positions_start = seq_positions - 1  # Now points to vision_start
#         selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#         # Compute SLVR loss between predicted and inserted slvr embeddings
#         loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Don't want CE loss for <slvr> token
#             shift_labels = shift_labels.masked_fill(shift_labels == self.config.slvr_id, IGNORE_INDEX)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss_ce = loss_fct(shift_logits, shift_labels)

#             # slvr loss
#             # Get last hidden states for <slvr> token positions
#             seq_positions_start = seq_positions - 1  # Now points to vision_start
#             selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#             # Compute SLVR loss between predicted and inserted slvr embeddings
#             loss_slvr = slvr_loss_fct(selected_hidden_states, selected_slvr_embeds)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         loss_ce=loss_ce,
#         loss_slvr=loss_slvr,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )
