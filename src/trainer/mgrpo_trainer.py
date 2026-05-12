"""
M-GRPO Trainer: Multi-query Group Relative Policy Optimization.

Key differences from QwenGRPO2QTrainer:
  1. Correct latent extraction:
     - visual latents  : hidden states at <|slvr|> token positions
                         (strictly between <|vision_start|> and the LAST <|vision_end|>)
     - semantic latent : hidden state at <|sem|> token position
                         (strictly between <sem> and </sem>)
     Missing tokens -> format penalty, 0-vector fallback for rewards.

  2. Latent consistency reward (R_cons):
       - lambda_sem * ||z_sem^(1) - z_sem^(2)||_2
       - lambda_vis * (1/T_v) * sum_t ||h_t^(1) - h_t^(2)||_2

  3. Stability reward (R_stab): margin-based against Stage-1 reference latents.

  4. Per-step logging: saves up to 10 reward cases to JSON for inspection.
"""

import os
import json
import re
import copy
import random
import torch
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union, Tuple

from accelerate.utils import gather, gather_object
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.models import unwrap_model_for_generation
from transformers import PreTrainedModel

from src.constants import MULTIMODAL_KEYWORDS
from qwen_vl_utils import process_vision_info
from src.train.monkey_patch_forward_slvr import replace_qwen2_5_with_mixed_modality_forward_slvr
from src.trainer.qq_grpo_trainer import (
    QwenGRPO2QTrainer,
    nanstd,
    nanmin,
    nanmax,
)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _extract_answer(text: str) -> str:
    """Extract <answer>...</answer> content if present; otherwise return empty."""
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def _extract_latents_from_hidden(
    hidden_states,   # (B, seq_len, H)
    token_ids,       # (B, seq_len)
    slvr_start_id, slvr_end_id, slvr_token_id,
    slvr_text_start_id, slvr_text_end_id, slvr_text_token_id,
):
    """
    Extract visual latents and semantic latent from last hidden states.

    Visual:  collect hidden[t] for ALL positions strictly between <|vision_start|>
             and the first <sem>.  In _slvr_deocding_with_latentend
             mode the token IDs in this span are all <|vision_end|> placeholders;
             the actual latent reasoning signal is entirely in the hidden states,
             so we collect every position regardless of its token ID.
    Semantic: collect hidden[t] for ALL positions strictly between
              <sem> and </sem> (or up to 8 positions if
              </sem> is absent).  Same rationale: token IDs are
              irrelevant, the hidden state carries the semantic representation.

    Returns:
        sem_latent  (B, H)          mean-pooled over the semantic span
        vis_latent  (B, T_max, H)   zero-padded
        vis_mask    (B, T_max) bool
        has_format  (B,) bool  -- True when BOTH spans are present
    """
    B, seq_len, H = hidden_states.shape
    device = hidden_states.device
    dtype  = hidden_states.dtype

    sem_list, vis_list = [], []
    has_format = []
    max_vis = 1

    for b in range(B):
        ids = token_ids[b]
        hs  = hidden_states[b]

        # ---- Locate span boundaries ----
        slvr_start_pos  = None
        slvr_ts_pos     = None   # <sem>
        slvr_te_pos     = None   # </sem>
        for t in range(seq_len):
            tok = ids[t].item()
            if tok == slvr_start_id and slvr_start_pos is None:
                slvr_start_pos = t
            if tok == slvr_text_start_id and slvr_ts_pos is None:
                slvr_ts_pos = t
            if tok == slvr_text_end_id and slvr_ts_pos is not None and slvr_te_pos is None:
                slvr_te_pos = t

        # ---- Visual latent: ALL hidden states in (vision_start, sem_start) ----
        # Token IDs in this span are placeholder <|vision_end|>s; we ignore them
        # and collect every position's last-layer hidden state.
        vis_hs = []
        if slvr_start_pos is not None and slvr_ts_pos is not None and slvr_ts_pos > slvr_start_pos + 1:
            for t in range(slvr_start_pos + 1, slvr_ts_pos):
                vis_hs.append(hs[t])

        # ---- Semantic latent: ALL hidden states in (sem_start, sem_end) ----
        # Mean-pool across the span to get a single (H,) vector.
        sem_hs = None
        if slvr_ts_pos is not None:
            te_bound = slvr_te_pos if slvr_te_pos is not None else min(slvr_ts_pos + 9, seq_len)
            if te_bound > slvr_ts_pos + 1:
                span = hs[slvr_ts_pos + 1 : te_bound]   # (T_sem, H)
                sem_hs = span.mean(dim=0)               # (H,)

        fmt_ok = len(vis_hs) > 0 and sem_hs is not None
        has_format.append(fmt_ok)

        sem_list.append(sem_hs if sem_hs is not None
                        else torch.zeros(H, device=device, dtype=dtype))

        if len(vis_hs) > 0:
            vt = torch.stack(vis_hs, dim=0)
            vis_list.append(vt)
            max_vis = max(max_vis, vt.size(0))
        else:
            vis_list.append(None)

    # Pad
    vis_latent = torch.zeros(B, max_vis, H, device=device, dtype=dtype)
    vis_mask   = torch.zeros(B, max_vis, device=device, dtype=torch.bool)
    for b in range(B):
        if vis_list[b] is not None:
            Tv = vis_list[b].size(0)
            vis_latent[b, :Tv] = vis_list[b]
            vis_mask[b, :Tv]   = True

    sem_latent  = torch.stack(sem_list, dim=0)
    has_format_t = torch.tensor(has_format, dtype=torch.bool, device=device)
    return sem_latent, vis_latent, vis_mask, has_format_t


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class QwenMGRPOTrainer(QwenGRPO2QTrainer):
    """Multi-query GRPO Trainer with latent consistency, stability, and reward logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_sem          = getattr(self.args, 'lambda_sem',          0.01)
        self.lambda_vis          = getattr(self.args, 'lambda_vis',          0.01)
        self.tau_sem             = getattr(self.args, 'tau_sem',             1.0)
        self.tau_vis             = getattr(self.args, 'tau_vis',             1.0)
        self.lambda_consistency  = getattr(self.args, 'lambda_consistency',  0.05)
        self.lambda_stability    = getattr(self.args, 'lambda_stability',    0.05)
        self.max_sem_pen         = float(os.getenv("MGRPO_MAX_SEM_PEN", "1.0"))
        self.max_vis_pen         = float(os.getenv("MGRPO_MAX_VIS_PEN", "1.0"))
        self._slvr_ids_cache = None

        # Reward log dir
        self._reward_log_dir = os.path.join(self.args.output_dir, "reward_logs")
        if self.accelerator.is_main_process:
            os.makedirs(self._reward_log_dir, exist_ok=True)

        # Per-step pure inference log dir (sampled from train set)
        self._infer_log_dir = os.path.join(self.args.output_dir, "inference_logs")
        self._infer_log_enabled = os.getenv("MGRPO_STEP_INFER_ENABLE", "1") == "1"
        self._infer_log_samples = max(1, int(os.getenv("MGRPO_STEP_INFER_SAMPLES", "5")))
        if self.accelerator.is_main_process:
            os.makedirs(self._infer_log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Token-ID helpers
    # ------------------------------------------------------------------

    def _get_slvr_ids(self):
        if self._slvr_ids_cache is None:
            cfg = self.accelerator.unwrap_model(self.model).config
            self._slvr_ids_cache = dict(
                slvr_start_id      = cfg.slvr_start_id,
                slvr_end_id        = cfg.slvr_end_id,
                slvr_token_id      = cfg.slvr_id,
                slvr_text_start_id = cfg.slvr_text_start_id,
                slvr_text_end_id   = cfg.slvr_text_end_id,
                slvr_text_token_id = cfg.slvr_text_id,
            )
        return self._slvr_ids_cache

    # ------------------------------------------------------------------
    # Latent extraction
    # ------------------------------------------------------------------

    def _extract_latents(self, model, prompt_completion_ids, attention_mask, multimodal_inputs):
        """Capture the final-layer hidden states via a forward hook on lm_head.

        Using output_hidden_states=True stores ALL 32 layer outputs (~10 GB for 7B),
        which causes CUDA memory pressure -> NCCL gradient all-reduce timeout.
        Hooking lm_head's input captures only the last (B, T, H) tensor (~300 MB).

        IMPORTANT: We always unwrap the model here so that DeepSpeed ZeRO-2's
        gradient hooks are NOT triggered.  Only the training forward in
        _compute_loss should go through the wrapped model.
        """
        raw_model = self.accelerator.unwrap_model(model)
        _captured = {}

        def _hook_fn(module, inp):
            # lm_head is a Linear; inp[0] is the (B, T, H) hidden-state tensor
            _captured['h'] = inp[0].detach()

        handle = raw_model.lm_head.register_forward_pre_hook(_hook_fn)
        try:
            with torch.no_grad():
                raw_model(
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    **multimodal_inputs,
                    return_dict=True,
                )
        finally:
            handle.remove()

        hidden = _captured.get('h')
        if hidden is None:
            raise RuntimeError("[M-GRPO] _extract_latents: lm_head hook did not fire")
        ids = self._get_slvr_ids()
        return _extract_latents_from_hidden(hidden, prompt_completion_ids, **ids)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _consistency_reward(self, sem1, sem2, vis1, vis2, vm1, vm2):
        # Use bounded cosine distance to keep this term numerically stable.
        sem_dist = 1.0 - F.cosine_similarity(sem1, sem2, dim=-1)
        L = min(vis1.size(1), vis2.size(1))
        shared = vm1[:, :L] & vm2[:, :L]
        tok_dist = 1.0 - F.cosine_similarity(vis1[:, :L], vis2[:, :L], dim=-1)
        vis_dist = (tok_dist * shared.float()).sum(-1) / shared.float().sum(-1).clamp(min=1.0)
        return -(self.lambda_sem * sem_dist + self.lambda_vis * vis_dist)

    def _stability_reward(self, sem, vis, vm, ref_sem, ref_vis, ref_vm):
        sem_dist = 1.0 - F.cosine_similarity(sem, ref_sem, dim=-1)
        sem_pen = torch.clamp(sem_dist - self.tau_sem, min=0.0, max=self.max_sem_pen)
        L = min(vis.size(1), ref_vis.size(1))
        shared = vm[:, :L] & ref_vm[:, :L]
        tok_dist = 1.0 - F.cosine_similarity(vis[:, :L], ref_vis[:, :L], dim=-1)
        vis_dist = (tok_dist * shared.float()).sum(-1) / shared.float().sum(-1).clamp(min=1.0)
        vis_pen = torch.clamp(vis_dist - self.tau_vis, min=0.0, max=self.max_vis_pen)
        return -(sem_pen + vis_pen)

    # ------------------------------------------------------------------
    # Per-step reward case logging
    # ------------------------------------------------------------------

    def _log_reward_cases(self, step, prompts_text, completions, assistants,
                          rewards_per_func, r_cons, r_stab, format_scores, final_rewards,
                          mm_debug=None, probe_info=None,
                          max_cases=10):
        if not self.accelerator.is_main_process:
            return
        cases = []
        n = min(len(completions), max_cases)
        for i in range(n):
            q1 = prompts_text[i * 2]     if i * 2     < len(prompts_text) else ""
            q2 = prompts_text[i * 2 + 1] if i * 2 + 1 < len(prompts_text) else ""
            comp = completions[i]
            sol  = assistants[i] if i < len(assistants) else {}
            cases.append({
                "pair_idx": i,
                "q1": q1, "q2": q2,
                "pred_q1": comp.get("text1-answer", ""),
                "pred_q2": comp.get("text2-answer", ""),
                "pred_answer_q1": _extract_answer(comp.get("text1-answer", "")),
                "pred_answer_q2": _extract_answer(comp.get("text2-answer", "")),
                "gt_q1":   sol.get("text1-answer", ""),
                "gt_q2":   sol.get("text2-answer", ""),
                "rewards_per_func": {
                    name: float(rewards_per_func[i, fi].item())
                    for fi, name in enumerate(self.reward_func_names)
                    if i < rewards_per_func.size(0)
                },
                "r_format":      float(format_scores[i].item()) if i < len(format_scores) else 0.0,
                "r_consistency": float(r_cons[i].item())         if i < len(r_cons) else 0.0,
                "r_stability":   float(r_stab[i].item())         if i < len(r_stab) else 0.0,
                "total_reward":  float(final_rewards[i].item())  if i < len(final_rewards) else 0.0,
                "multimodal_debug": mm_debug or {},
                "probe_inference": probe_info or {},
            })
        out = os.path.join(self._reward_log_dir, f"step_{step:06d}.json")
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MGRPO] reward log write error: {e}", flush=True)

    def _build_infer_messages_from_prompt(self, prompt):
        """Build user-only chat messages (same style as old inference script)."""
        content = None
        for msg in prompt:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                break
        if content is None and len(prompt) > 0:
            content = prompt[0].get("content", [])

        vision_content = []
        for item in (content or []):
            if item.get("type") in {"image", "video"}:
                vision_content.append(copy.deepcopy(item))

        out = []
        for item in (content or []):
            t = item.get("type", "")
            if t == "text1":
                q = item.get("text1", "")
                out.append((
                    "text1",
                    q,
                    [{"role": "user", "content": [*copy.deepcopy(vision_content), {"type": "text", "text": q}]}],
                ))
            elif t == "text2":
                q = item.get("text2", "")
                out.append((
                    "text2",
                    q,
                    [{"role": "user", "content": [*copy.deepcopy(vision_content), {"type": "text", "text": q}]}],
                ))
        return out

    def _log_step_inference_cases(self, step):
        """Run pure inference every step on random train samples and save outputs."""
        if not self.accelerator.is_main_process or not self._infer_log_enabled:
            return
        if self.train_dataset is None:
            return

        try:
            ds_len = len(self.train_dataset)
        except Exception:
            return
        if ds_len <= 0:
            return

        n = min(self._infer_log_samples, ds_len)
        sampled_indices = random.sample(range(ds_len), n)

        sampled_items = []
        for idx in sampled_indices:
            try:
                sampled_items.append((idx, self.train_dataset[idx]))
            except Exception:
                continue
        if not sampled_items:
            return

        batch_messages = []
        meta = []
        for ds_idx, item in sampled_items:
            prompt = item.get("prompt", [])
            assistant = item.get("assistant", {})
            for q_key, question, messages in self._build_infer_messages_from_prompt(prompt):
                batch_messages.append(messages)
                gt_key = f"{q_key}-answer"
                meta.append({
                    "dataset_index": ds_idx,
                    "question_key": q_key,
                    "question": question,
                    "ground_truth": assistant.get(gt_key, ""),
                })

        if not batch_messages:
            return

        prompts_text = [
            self.processing_class.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        infer_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **video_kwargs,
        )
        infer_inputs = super(QwenGRPO2QTrainer, self)._prepare_inputs(infer_inputs)

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            model_cfg = unwrapped_model.config
            slvr_head_enabled = bool(getattr(model_cfg, "slvr_head", False))
            latent_end_enabled = bool(getattr(model_cfg, "latent_end_token", False))
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            replace_qwen2_5_with_mixed_modality_forward_slvr(
                inference_mode=True,
                slvr_head=slvr_head_enabled,
            )
            try:
                with torch.no_grad():
                    infer_ids = unwrapped_model.generate(
                        **infer_inputs,
                        generation_config=self.generation_config,
                    )
            finally:
                replace_qwen2_5_with_mixed_modality_forward_slvr(
                    inference_mode=False,
                    rl=True,
                    slvr_head=slvr_head_enabled,
                    latent_end_token=latent_end_enabled,
                )
                if was_training:
                    unwrapped_model.train()

        infer_trim = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(infer_inputs["input_ids"], infer_ids)
        ]
        infer_texts = self.processing_class.batch_decode(
            infer_trim,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        rows = []
        for m, pred in zip(meta, infer_texts):
            rows.append({
                **m,
                "prediction": pred,
                "pred_answer": _extract_answer(pred),
                "has_answer_tag": ("<answer>" in pred and "</answer>" in pred),
            })

        out = os.path.join(self._infer_log_dir, f"step_{step:06d}.json")
        payload = {
            "step": int(step),
            "num_sampled_items": len(sampled_items),
            "num_infer_queries": len(rows),
            "sampled_indices": sampled_indices,
            "decoding_strategy": getattr(self.generation_config, "decoding_strategy", None),
            "slvr_steps": getattr(self.generation_config, "slvr_steps", None),
            "records": rows,
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MGRPO] inference log write error: {e}", flush=True)

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"
        num_questions = 2
        self._get_slvr_ids()
        slvr_ids = self._slvr_ids_cache

        # ---- Build prompts ----
        prompts = [x["prompt"] for x in inputs]
        batch_messages = []
        prompts_text = []
        for prompt in prompts:
            content = None
            for msg in prompt:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
            if content is None and len(prompt) > 0:
                content = prompt[0].get("content", [])

            vision_content = []
            for item in content:
                if item.get("type") in {"image", "video"}:
                    vision_content.append(copy.deepcopy(item))

            for item in content:
                t = item.get("type", "")
                if t == "text1":
                    question = item.get("text1", "")
                    messages = [{
                        "role": "user",
                        "content": [*copy.deepcopy(vision_content), {"type": "text", "text": question}],
                    }]
                    batch_messages.append(messages)
                    prompts_text.append(
                        self.processing_class.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                elif t == "text2":
                    question = item.get("text2", "")
                    messages = [{
                        "role": "user",
                        "content": [*copy.deepcopy(vision_content), {"type": "text", "text": question}],
                    }]
                    batch_messages.append(messages)
                    prompts_text.append(
                        self.processing_class.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

        # Vision
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages, return_video_kwargs=True
        )
        prompt_inputs = self.processing_class(
            text=prompts_text, images=image_inputs, videos=video_inputs,
            padding=True, padding_side="left", return_tensors="pt", **video_kwargs,
        )
        prompt_inputs = super(QwenGRPO2QTrainer, self)._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        has_multimodal_prompt = (
            ("pixel_values" in prompt_inputs and prompt_inputs.get("pixel_values") is not None)
            or ("pixel_values_videos" in prompt_inputs and prompt_inputs.get("pixel_values_videos") is not None)
        )
        if self.max_prompt_length is not None and not has_multimodal_prompt:
            prompt_ids  = prompt_ids[:,  -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ---- Generate ----
        mm_debug = {
            "num_prompts_text": len(prompts_text),
            "num_image_inputs": len(image_inputs) if isinstance(image_inputs, list) else 0,
            "num_video_inputs": len(video_inputs) if isinstance(video_inputs, list) else 0,
            "has_pixel_values": "pixel_values" in prompt_inputs and prompt_inputs.get("pixel_values") is not None,
            "has_image_grid_thw": "image_grid_thw" in prompt_inputs and prompt_inputs.get("image_grid_thw") is not None,
            "has_pixel_values_videos": "pixel_values_videos" in prompt_inputs and prompt_inputs.get("pixel_values_videos") is not None,
            "has_video_grid_thw": "video_grid_thw" in prompt_inputs and prompt_inputs.get("video_grid_thw") is not None,
            "pixel_values_shape": list(prompt_inputs["pixel_values"].shape) if "pixel_values" in prompt_inputs and prompt_inputs.get("pixel_values") is not None else None,
            "image_grid_thw_shape": list(prompt_inputs["image_grid_thw"].shape) if "image_grid_thw" in prompt_inputs and prompt_inputs.get("image_grid_thw") is not None else None,
        }

        probe_info = {}
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            model_cfg = unwrapped_model.config
            slvr_head_enabled = bool(getattr(model_cfg, "slvr_head", False))
            latent_end_enabled = bool(getattr(model_cfg, "latent_end_token", False))
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            # Keep rollout generation path identical to old inference behavior.
            replace_qwen2_5_with_mixed_modality_forward_slvr(
                inference_mode=True,
                slvr_head=slvr_head_enabled,
            )
            try:
                with torch.no_grad():
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=self.generation_config
                    )

                # Per-step probe inference on one training sample (main process only)
                if self.accelerator.is_main_process and len(prompts_text) > 0:
                    try:
                        probe_messages = batch_messages[0]
                        probe_image_inputs, probe_video_inputs, probe_video_kwargs = process_vision_info(
                            [probe_messages], return_video_kwargs=True
                        )
                        probe_kwargs = {
                            "text": [prompts_text[0]],
                            "padding": True,
                            "padding_side": "left",
                            "return_tensors": "pt",
                        }
                        if probe_image_inputs is not None:
                            probe_kwargs["images"] = probe_image_inputs
                        if probe_video_inputs is not None:
                            probe_kwargs["videos"] = probe_video_inputs
                        probe_kwargs.update(probe_video_kwargs)

                        probe_inputs = self.processing_class(**probe_kwargs)
                        probe_inputs = super(QwenGRPO2QTrainer, self)._prepare_inputs(probe_inputs)

                        with torch.no_grad():
                            probe_ids = unwrapped_model.generate(
                                **probe_inputs,
                                max_new_tokens=min(128, self.max_completion_length),
                                do_sample=False,
                                eos_token_id=self.processing_class.tokenizer.eos_token_id,
                                pad_token_id=self.processing_class.tokenizer.pad_token_id,
                                decoding_strategy=getattr(self.generation_config, "decoding_strategy", "latent"),
                                slvr_steps=[getattr(self.generation_config, "slvr_steps", 8)],
                                criterion=getattr(self.generation_config, "criterion", "mse"),
                                slvr_end_threshold=getattr(self.generation_config, "slvr_end_threshold", 0.02),
                            )
                        probe_trim = [
                            out_ids[len(in_ids):]
                            for in_ids, out_ids in zip(probe_inputs["input_ids"], probe_ids)
                        ]
                        probe_text = self.processing_class.batch_decode(
                            probe_trim, skip_special_tokens=False, clean_up_tokenization_spaces=True
                        )[0]
                        probe_info = {
                            "ok": True,
                            "prompt_preview": prompts_text[0][:200],
                            "output": probe_text,
                            "output_has_answer_tag": ("<answer>" in probe_text and "</answer>" in probe_text),
                        }
                    except Exception as e:
                        probe_info = {"ok": False, "error": str(e)}
            finally:
                replace_qwen2_5_with_mixed_modality_forward_slvr(
                    inference_mode=False,
                    rl=True,
                    slvr_head=slvr_head_enabled,
                    latent_end_token=latent_end_enabled,
                )
                if was_training:
                    unwrapped_model.train()

        # Extract multimodal inputs before freeing prompt_inputs
        multimodal_inputs = {k: prompt_inputs.get(k) for k in MULTIMODAL_KEYWORDS}

        # Free KV-cache memory from generation before scoring/loss
        del prompt_inputs
        torch.cuda.empty_cache()

        prompt_length = prompt_ids.size(1)
        prompt_ids    = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # EOS mask
        is_eos  = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(1)] = is_eos.int().argmax(1)[is_eos.any(1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        if self.mask_truncated_completions:
            completion_mask = completion_mask * (~(~is_eos.any(1))).unsqueeze(1).int()

        # Mask SLVR+SLVR_text spans from log-prob loss
        slvr_token_mask = torch.ones_like(prompt_completion_ids, dtype=torch.bool)
        for b in range(prompt_completion_ids.size(0)):
            in_vis  = False
            in_text = False
            for t in range(prompt_completion_ids.size(1)):
                tok = prompt_completion_ids[b, t].item()
                if tok == slvr_ids['slvr_start_id']:      in_vis  = True
                elif tok == slvr_ids['slvr_end_id']:      in_vis  = False
                if tok == slvr_ids['slvr_text_start_id']: in_text = True
                elif tok == slvr_ids['slvr_text_end_id']: in_text = False
                if in_vis or in_text:
                    slvr_token_mask[b, t] = False

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        final_mask     = attention_mask.bool() & slvr_token_mask

        logits_to_keep = completion_ids.size(1)

        # Log-probs
        def _score(model, ids, mask, ltk, **kw):
            if getattr(self.generation_config, 'slvr_steps', 0) > 0:
                try:
                    return self.score_with_slvr_replay(
                        model,
                        input_ids=ids[:, :prompt_ids.size(1)],
                        attention_mask=mask[:, :prompt_ids.size(1)],
                        target_ids=ids,
                        generation_config=self.generation_config,
                        slvr_steps=[self.generation_config.slvr_steps] * ids.size(0),
                        enable_grad=False,
                        **kw,
                    )
                except RuntimeError as e:
                    err = str(e)
                    err_l = err.lower()
                    is_known_rope_mismatch = (
                        "shape mismatch" in err_l
                        and (
                            "cannot be broadcast" in err_l
                            or "broadcast to indexing result" in err_l
                            or "indexing result of shape" in err_l
                        )
                    )
                    if not is_known_rope_mismatch:
                        raise

                    if self.accelerator.is_main_process:
                        self.accelerator.print(
                            "[MGRPO][warn] score_with_slvr_replay hit Qwen-VL rope shape mismatch; "
                            "falling back to _get_per_token_logps for this step. "
                            f"error={err[:220]}"
                        )
                    return self._get_per_token_logps(model, ids, mask, ltk, **kw)
            return self._get_per_token_logps(model, ids, mask, ltk, **kw)

        # IMPORTANT: Use unwrapped models for all no-grad scoring forwards.
        # Calling forward on the DeepSpeed-wrapped model under torch.no_grad()
        # confuses ZeRO-2 gradient bucket accounting and causes NCCL deadlocks.
        _unwrapped_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            old_per_token_logps = None
            if self.num_iterations > 1:
                old_per_token_logps = _score(_unwrapped_model, prompt_completion_ids, attention_mask, logits_to_keep, **multimodal_inputs)
                old_per_token_logps = old_per_token_logps * final_mask[:, -logits_to_keep:]
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                _unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
                ref_per_token_logps = _score(_unwrapped_ref, prompt_completion_ids, attention_mask, logits_to_keep, **multimodal_inputs)
                ref_per_token_logps = ref_per_token_logps * final_mask[:, -logits_to_keep:]
            else:
                with _unwrapped_model.disable_adapter():
                    ref_per_token_logps = _score(_unwrapped_model, prompt_completion_ids, attention_mask, logits_to_keep, **multimodal_inputs)

        # ---- Decode completions ----
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        completions = []
        for i in range(0, len(completions_text), num_questions):
            completions.append({
                "text1-answer": completions_text[i],
                "text2-answer": completions_text[i + 1] if i + 1 < len(completions_text) else "",
            })

        # ---- External rewards ----
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        format_scores    = torch.zeros(len(completions), device=device)
        for fi, (rfn, rpc, rname) in enumerate(zip(
            self.reward_funcs, self.reward_processing_classes, self.reward_func_names
        )):
            with profiling_context(self, rname):
                keys = [k for k in inputs[0] if k not in ("prompt", "completion")]
                rkwargs = {k: [ex[k] for ex in inputs] for k in keys}
                out = rfn(prompts=inputs, completions=completions, **rkwargs)
                out = [r if r is not None else float('nan') for r in out]
                rewards_per_func[:, fi] = torch.tensor(out, dtype=torch.float32, device=device)
                if "format" in rname.lower():
                    format_scores = rewards_per_func[:, fi].clone()

        # ---- Latent extraction ----
        # IMPORTANT: use the UNWRAPPED model so DeepSpeed ZeRO-2 gradient hooks
        # are not triggered — avoids NCCL allreduce deadlock.
        B_exp = prompt_completion_ids.size(0)
        sem, vis, vm, has_fmt = self._extract_latents(
            _unwrapped_model, prompt_completion_ids, attention_mask, multimodal_inputs
        )

        q1_idx = list(range(0, B_exp, 2))
        q2_idx = list(range(1, B_exp, 2))
        pair_has_fmt = has_fmt[q1_idx] & has_fmt[q2_idx]

        r_cons = self._consistency_reward(
            sem[q1_idx], sem[q2_idx], vis[q1_idx], vis[q2_idx], vm[q1_idx], vm[q2_idx]
        )

        if self.ref_model is not None and self.lambda_stability > 0:
            ref_u = self.accelerator.unwrap_model(self.ref_model)
            ref_sem, ref_vis, ref_vm, _ = self._extract_latents(
                ref_u, prompt_completion_ids, attention_mask, multimodal_inputs
            )
            r_stab = (
                self._stability_reward(sem[q1_idx], vis[q1_idx], vm[q1_idx],
                                       ref_sem[q1_idx], ref_vis[q1_idx], ref_vm[q1_idx])
                + self._stability_reward(sem[q2_idx], vis[q2_idx], vm[q2_idx],
                                         ref_sem[q2_idx], ref_vis[q2_idx], ref_vm[q2_idx])
            ) / 2.0
        else:
            r_stab = torch.zeros(len(completions), device=device)

        # ---- Gate rewards on format / extractability ----
        # If a sample has no valid SLVR format (format_score <= 0), the model
        # cannot meaningfully extract a structured answer, so zero out its
        # accuracy reward.  This makes format a hard prerequisite for earning
        # accuracy signal and prevents the bypass strategy from being rewarded.
        # Additionally, only compute latent consistency/stability when both
        # query completions have extractable latent spans.
        no_format_mask = (format_scores <= 0)   # (N_completions,)
        for fi, rname in enumerate(self.reward_func_names):
            if "accuracy" in rname.lower():
                rewards_per_func[:, fi] = rewards_per_func[:, fi] * (~no_format_mask).float()

        valid_latent_mask = (~no_format_mask) & pair_has_fmt
        r_cons = torch.where(valid_latent_mask, r_cons, torch.zeros_like(r_cons))
        r_stab = torch.where(valid_latent_mask, r_stab, torch.zeros_like(r_stab))

        # ---- Combine ----
        rpf_gathered = gather(rewards_per_func)
        rewards      = (rpf_gathered * self.reward_weights.to(device).unsqueeze(0)).nansum(1)
        rc_gathered  = gather(r_cons)
        rs_gathered  = gather(r_stab)
        rewards      = rewards + self.lambda_consistency * rc_gathered + self.lambda_stability * rs_gathered

        # ---- Advantages ----
        mg = rewards.view(-1, self.num_generations).mean(1)
        sg = rewards.view(-1, self.num_generations).std(1)
        mg = mg.repeat_interleave(self.num_generations, 0)
        sg = sg.repeat_interleave(self.num_generations, 0)
        adv = rewards - mg
        if self.scale_rewards:
            adv_norm_eps = float(os.getenv("MGRPO_ADV_NORM_EPS", "1e-3"))
            adv = adv / (sg + max(adv_norm_eps, 1e-8))
        adv_clip = float(os.getenv("MGRPO_ADV_CLIP", "5.0"))
        if adv_clip > 0:
            adv = torch.clamp(adv, min=-adv_clip, max=adv_clip)
        pslice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        adv = adv[pslice]
        adv_expanded = adv.repeat_interleave(num_questions)

        # ---- Per-step logging ----
        assistants    = [x.get("assistant", {}) for x in inputs]
        local_rpf     = rpf_gathered[pslice]
        local_rewards = rewards[pslice]
        self._log_reward_cases(
            step=self.state.global_step,
            prompts_text=prompts_text,
            completions=completions,
            assistants=assistants,
            rewards_per_func=local_rpf,
            r_cons=r_cons,
            r_stab=r_stab,
            format_scores=format_scores,
            final_rewards=local_rewards,
            mm_debug=mm_debug,
            probe_info=probe_info,
        )
        self._log_step_inference_cases(step=self.state.global_step)

        # ---- Metrics ----
        disable_metric_gather = os.getenv("MGRPO_DISABLE_METRIC_GATHER", "0") == "1"
        if mode == "train":
            if disable_metric_gather:
                self.state.num_input_tokens_seen += attention_mask.sum().item()
            else:
                self.state.num_input_tokens_seen += (
                    self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
                )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        acm = completion_mask.sum(1) if disable_metric_gather else self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(acm.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(acm.float().min().item())
        self._metrics[mode]["completions/max_length"].append(acm.float().max().item())
        aeos = is_eos.any(1) if disable_metric_gather else self.accelerator.gather_for_metrics(is_eos.any(1))
        tcm  = acm[aeos]
        self._metrics[mode]["completions/clipped_ratio"].append(1 - len(tcm) / len(acm))
        if len(tcm) == 0:
            tcm = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(tcm.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(tcm.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(tcm.float().max().item())
        for fi, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rpf_gathered[:, fi]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(rpf_gathered[:, fi]).item())
        self._metrics[mode]["reward"].append(mg.mean().item())
        self._metrics[mode]["reward_std"].append(sg.mean().item())
        self._metrics[mode]["reward_consistency"].append(rc_gathered.mean().item())
        self._metrics[mode]["reward_stability"].append(rs_gathered.mean().item())

        if disable_metric_gather:
            self._textual_logs["prompt"].extend(prompts_text)
            self._textual_logs["completion"].extend(completions_text)
        else:
            self._textual_logs["prompt"].extend(gather_object(prompts_text))
            self._textual_logs["completion"].extend(gather_object(completions_text))
        for fi, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rpf_gathered[:, fi].tolist())

        return {
            "prompt_ids":           prompt_ids,
            "prompt_mask":          prompt_mask,
            "completion_ids":       completion_ids,
            "completion_mask":      completion_mask,
            "final_mask":           final_mask[:, -logits_to_keep:],
            "advantages":           adv_expanded,
            "old_per_token_logps":  old_per_token_logps,
            "ref_per_token_logps":  ref_per_token_logps,
            "multimodal_inputs":    multimodal_inputs,
        }
