import argparse
import json
import os
import random
import re
from typing import List, Dict, Any

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from qwen_vl_utils import process_vision_info
from src.model.qwen_slvr_model import QwenWithSLVR
from src.train.monkey_patch_forward_slvr import replace_qwen2_5_with_mixed_modality_forward_slvr


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def replace_image_tokens(text: str) -> str:
    return re.sub(r"\n?" + re.escape("<image>") + r"\n?", "", text).strip()


def ensure_slvr_tokens_and_ids(model, processor):
    tokenizer = processor.tokenizer
    token_pairs = [
        ("slvr_start_id", "<|vision_start|>"),
        ("slvr_id", "<|slvr|>"),
        ("slvr_latent_end_id", "<|slvr_latent_end|>"),
        ("slvr_end_id", "<|vision_end|>"),
        ("slvr_text_start_id", "<sem>"),
        ("slvr_text_id", "<|sem|>"),
        ("slvr_text_latent_end_id", "<|sem_latent_end|>"),
        ("slvr_text_end_id", "</sem>"),
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


def load_pairs(data_path: str) -> List[Any]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert len(raw) % 2 == 0, f"Dataset length {len(raw)} is not even"
    return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]


def resolve_images(item: Dict[str, Any], image_folder: str) -> List[str]:
    images = item.get("images", [])
    if not isinstance(images, list):
        images = [images]
    out = []
    for p in images:
        if isinstance(p, str) and p and (p.startswith("http") or os.path.exists(p)):
            out.append(p)
        else:
            out.append(os.path.join(image_folder, p))
    return out


def build_messages(images: List[str], question: str):
    content = [{"type": "image", "image": p} for p in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--decoding_strategy", default="latent")
    parser.add_argument("--slvr_steps", type=int, default=8)
    parser.add_argument("--criterion", default="mse")
    parser.add_argument("--slvr_end_threshold", type=float, default=0.02)
    parser.add_argument("--latent_end_token", default="true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(args.model_pth, trust_remote_code=True)
    if str2bool(args.latent_end_token):
        config.latent_end_token = True

    replace_qwen2_5_with_mixed_modality_forward_slvr(
        inference_mode=True,
        slvr_head=bool(getattr(config, "slvr_head", False)),
    )

    model = QwenWithSLVR.from_pretrained(
        args.model_pth,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_pth, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    added = ensure_slvr_tokens_and_ids(model, processor)

    pairs = load_pairs(args.data_path)
    chosen = random.sample(range(len(pairs)), min(args.num_samples, len(pairs)))

    batch_messages = []
    meta = []

    for pair_idx in chosen:
        item1, item2 = pairs[pair_idx]
        images = resolve_images(item1, args.image_folder)

        q1 = replace_image_tokens(item1["conversations"][0]["value"])
        q2 = replace_image_tokens(item2["conversations"][0]["value"])

        m1 = build_messages(images, q1)
        m2 = build_messages(images, q2)

        batch_messages.extend([m1, m2])
        meta.extend([
            {
                "pair_idx": pair_idx,
                "question_key": "text1",
                "question": q1,
                "gt": item1["conversations"][1]["value"],
            },
            {
                "pair_idx": pair_idx,
                "question_key": "text2",
                "question": q2,
                "gt": item2["conversations"][1]["value"],
            },
        ])

    prompts_text = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]

    image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
    model_inputs = processor(
        text=prompts_text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoding_strategy=args.decoding_strategy,
        slvr_steps=args.slvr_steps,
        criterion=args.criterion,
        slvr_end_threshold=args.slvr_end_threshold,
    )

    with torch.no_grad():
        out_ids = model.generate(**model_inputs, generation_config=gen_cfg)

    trim_ids = [
        out[len(inp):]
        for inp, out in zip(model_inputs["input_ids"], out_ids)
    ]
    preds = processor.batch_decode(trim_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

    records = []
    answer_tag_hits = 0
    for m, pred in zip(meta, preds):
        has_tag = ("<answer>" in pred and "</answer>" in pred)
        answer_tag_hits += int(has_tag)
        records.append({
            **m,
            "prediction": pred,
            "has_answer_tag": has_tag,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "preflight_inference.json")
    payload = {
        "model_pth": args.model_pth,
        "config": {
            "slvr_head": getattr(config, "slvr_head", None),
            "latent_end_token": getattr(config, "latent_end_token", None),
            "added_special_tokens": added,
        },
        "generation": {
            "decoding_strategy": args.decoding_strategy,
            "slvr_steps": args.slvr_steps,
            "criterion": args.criterion,
            "slvr_end_threshold": args.slvr_end_threshold,
            "do_sample": False,
        },
        "num_records": len(records),
        "answer_tag_ratio": answer_tag_hits / max(len(records), 1),
        "records": records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[preflight] saved: {out_path}")
    print(f"[preflight] answer_tag_ratio={payload['answer_tag_ratio']:.3f} ({answer_tag_hits}/{len(records)})")


if __name__ == "__main__":
    main()
