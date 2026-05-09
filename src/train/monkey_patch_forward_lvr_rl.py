import torch
import os
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
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


def replace_qwen2_5_with_mixed_modality_forward_lvr_rl():
    verbose = os.getenv("MGRPO_FORWARD_PATCH_VERBOSE", "0") == "1"
    if verbose:
        print("This forward function is seperated from the others as SFT and RL stage have different version of transformers. This will be fixed later.")
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_grpo


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
    loss_lvr_text: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mode_switch: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # next_pos_lvr:Optional[bool] = False


def  set_lvr_loss_fct(loss_lvr_fct: str):
    """
        Set the loss function for LVR.
        Args:
            loss_lvr_fct (str): The type of loss function to use for LVR.
        Returns:
            A loss function object.
    """
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")


"""
    Forward function for stage 2 RL
    Kinda messy since in this stage, the transofmers will be 4.51.3 < 4.54 in stage I
    Will fix this inconsistency in final release
"""
def qwen2_5_mixed_modality_forward_lvr_grpo(
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
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for GENERATION: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for GENERATION: last hidden state of the last position
    lvr_mask: Optional[torch.FloatTensor] = None,   # This is for RL loss computation
    lvr_states: Optional[torch.FloatTensor] = None, # This is for RL loss computation
    prompt_length: Optional[int] = None, # This is for RL loss computation
    text_embedding: Optional[torch.FloatTensor] = None,  # 新增：text embedding参数
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''In this mode, no lvr_tokens'''
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        Generation: inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        Teacher-forcing fwd pass: patch lvr states
    '''
    if lvr_states is not None and lvr_mask is not None:
        comp_embeds = inputs_embeds[:, prompt_length:, :]  # (B, C, H)
        comp_embeds = torch.where(
            lvr_mask.unsqueeze(-1),   # (B, C, 1)
            lvr_states,               # (B, C, H)
            comp_embeds               # (B, C, H)
        )
        inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length, :], comp_embeds], dim=1)
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed
    # edge cases when a batch has text-only inputs during training.
    if lvr_mode_switch is None and (pixel_values is None and pixel_values_videos is None):
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)

        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
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
    
    # 处理text_embedding
    projected_text_embeddings = None
    if text_embedding is not None and hasattr(self.config, 'lvr_text_id'):
        # 1. 识别<|sem|> token的位置
        lvr_text_mask = input_ids == self.config.lvr_text_id
        batch_indices_text, seq_positions_text = torch.nonzero(lvr_text_mask, as_tuple=True)
        
        if lvr_text_mask.any():
            batch_size = input_ids.size(0)
            
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
            # 注意：这里需要确保模型有text_embedding_projector属性
            if hasattr(self, 'text_embedding_projector'):
                projected_text_embeddings = self.text_embedding_projector(text_embedding)
            else:
                # 如果没有投影层，可能需要创建一个
                # 这里假设hidden_size可以从inputs_embeds获取
                hidden_size = inputs_embeds.shape[-1]
                projected_text_embeddings = torch.nn.functional.linear(
                    text_embedding, 
                    torch.eye(4096, hidden_size).to(text_embedding.device)
                )
            
            # 替换embeddings
            inputs_embeds[batch_indices_text, seq_positions_text] = projected_text_embeddings[batch_indices_text].to(inputs_embeds.device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
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
        **kwargs,
    )

    # check if there is lvr_head
    if self.config.lvr_head:
        '''apply lvr_head in _inference mode'''
        if lvr_mode_switch is not None:
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    # ------------------------------------------------------------------
    # ZeRO-2 deadlock prevention: ensure conditionally-unused modules
    # always participate in the backward graph.  Without this, their
    # gradient hooks never fire and ZeRO-2 deadlocks waiting for the
    # corresponding allreduce.
    # Pattern: (x - x.detach()) adds exactly 0 to logits but keeps the
    # full backward graph so that parameter gradients (all zeros) are
    # computed and hooks fire.
    # ------------------------------------------------------------------
    if self.config.lvr_head and lvr_mode_switch is None:
        _lvr_d = self.lvr_head(hidden_states[:1, :1, :].detach()).sum()
        logits = logits + (_lvr_d - _lvr_d.detach())
    if text_embedding is None and hasattr(self, 'text_embedding_projector'):
        _te_inp = torch.zeros(1, self.text_embedding_projector.in_features if hasattr(self.text_embedding_projector, 'in_features') else 4096,
                              device=logits.device, dtype=logits.dtype)
        _te_d = self.text_embedding_projector(_te_inp).sum()
        logits = logits + (_te_d - _te_d.detach())

    # 设置损失函数
    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    lvr_text_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    loss = None
    loss_ce = None
    loss_lvr = None
    loss_lvr_text = None  # 新增：text embedding损失
    
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
        # Don't want CE loss for <lvr> token 和 <sem_placeholder> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_text_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # No lvr loss in this mode (除非有lvr_states)
        loss_lvr = None
        
        # 计算sem损失
        if text_embedding is not None and hasattr(self.config, 'lvr_text_id') and projected_text_embeddings is not None:
            # 重新识别<|sem|>的位置
            lvr_text_mask = input_ids == self.config.lvr_text_id
            batch_indices_text, seq_positions_text = torch.nonzero(lvr_text_mask, as_tuple=True)
            
            if lvr_text_mask.any():
                # 获取<sem>位置（<|sem|>的前一个位置）
                seq_positions_text_start = seq_positions_text - 1
                
                # 获取这些位置的hidden states
                selected_hidden_states_text = hidden_states[batch_indices_text, seq_positions_text_start].to(torch.float32)
                
                # 确保projected_text_embeddings的形状正确
                # 需要根据batch_indices_text选择对应的嵌入
                target_text_embeddings = projected_text_embeddings[batch_indices_text].to(torch.float32)
                
                loss_lvr_text = lvr_text_loss_fct(selected_hidden_states_text, target_text_embeddings)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_lvr_text=loss_lvr_text,  # 新增：返回text embedding损失
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state
    )