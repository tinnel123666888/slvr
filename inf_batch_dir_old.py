import sys
import os
import shutil

# =======================
# 仅新增：选卡（必须在 import torch 前）
# =======================
GPU_IDS = os.getenv("GPU_IDS", "")
if GPU_IDS:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

import torch
import json
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import socket

from src.model.qwen_slvr_model import QwenWithSLVR
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
class Config:
    STEP = "6287"
    MODEL_PATH = f"./stage1_checkpoints/checkpoint-{STEP}"

    # <<< changed >>> 改成目录
    INPUT_DIR = "/path/to/your/test_dataset"

    ckpt_dir = os.path.dirname(MODEL_PATH)
    ckpt_name = os.path.basename(ckpt_dir)
    OUTPUT_DIR = os.path.join("./old_test", ckpt_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # <<< changed >>> 每个json单独保存，输出根目录固定到这里
    OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, f"test_sslvr_results_{STEP}")
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

    # 推理参数
    STEPS = 8
    DECODING_STRATEGY = "latent"
    TASK_INSTRUCTION = ""

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
IS_DISTRIBUTED = WORLD_SIZE > 1
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")


def set_runtime(rank: int, local_rank: int, world_size: int):
    global RANK, LOCAL_RANK, WORLD_SIZE, IS_DISTRIBUTED, DEVICE
    RANK = rank
    LOCAL_RANK = local_rank
    WORLD_SIZE = world_size
    IS_DISTRIBUTED = WORLD_SIZE > 1
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")


def init_distributed_if_needed():
    if IS_DISTRIBUTED and not dist.is_initialized():
        init_kwargs = dict(
            backend="nccl",
            rank=RANK,
            world_size=WORLD_SIZE,
            init_method="env://",
        )
        # 新版 torch 支持 device_id，可避免 NCCL 对 rank->device 的猜测告警。
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")

        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            # 兼容不支持 device_id 参数的 torch 版本。
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)


def distributed_barrier_if_needed():
    if not dist.is_initialized():
        return
    try:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[LOCAL_RANK])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


def cleanup_distributed_if_needed():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(msg: str):
    if RANK == 0:
        print(msg)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _distributed_worker(local_rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    set_runtime(rank=local_rank, local_rank=local_rank, world_size=world_size)
    main()


def auto_launch_multi_gpu_if_needed() -> bool:
    # 支持手动关闭自动多卡：AUTO_MULTI_GPU=0
    if os.getenv("AUTO_MULTI_GPU", "1") in {"0", "false", "False"}:
        return False

    # torchrun / 手工设置了 WORLD_SIZE 时，不做自动拉起
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        return False

    if not torch.cuda.is_available():
        return False

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return False

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(_find_free_port()))
    print(f"[Launcher] 检测到 {world_size} 张可见GPU，自动启用多卡分布式推理")
    print(f"[Launcher] MASTER_ADDR={master_addr} MASTER_PORT={master_port}")

    mp.spawn(
        _distributed_worker,
        nprocs=world_size,
        args=(world_size, master_addr, master_port),
        join=True,
    )
    return True

# ==========================================
# 2. 原始核心函数 (保持不变)
# ==========================================
def load_model_and_processor(chkpt_pth):
    """加载模型和处理器"""
    config = AutoConfig.from_pretrained(chkpt_pth)

    from src.train.monkey_patch_forward_slvr import replace_qwen2_5_with_mixed_modality_forward_slvr
    replace_qwen2_5_with_mixed_modality_forward_slvr(
        inference_mode=True,
        slvr_head=config.slvr_head
    )
    config.latent_end_token = True

    model = QwenWithSLVR.from_pretrained(
        chkpt_pth,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        # flash_attn 在当前环境存在二进制符号不兼容，改用更稳的 SDPA
        attn_implementation="sdpa",
        device_map={"": LOCAL_RANK} if torch.cuda.is_available() else "cpu",
    )

    processor = AutoProcessor.from_pretrained(chkpt_pth)
    processor.tokenizer.padding_side = "left"
    return model, processor


def create_messages(img_path, question):
    """创建对话消息格式"""
    if not isinstance(img_path, list):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question},
            ],
        }]
    else:
        vision_content = [{"type": "image", "image": ip} for ip in img_path]
        vision_content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": vision_content}]
    return messages


def run_single_inference(model, processor, img_path, question, steps, decoding_strategy):
    """执行单张图片推理"""
    messages = create_messages(img_path, question)
    text_formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            slvr_steps=[steps],
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

    # 清理多余的 endoftext 标记
    text = output_text[0]
    return text


def extract_answer(response):
    """从模型响应中提取答案"""
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + len("<answer>")
        end = response.find("</answer>")
        if end > start:
            return response[start:end].strip()

    if "</answer>" in response:
        parts = response.split("</answer>")
        last_part = parts[-2] if len(parts) > 1 else parts[0]
        if "<answer>" in last_part:
            return last_part.split("<answer>")[-1].strip()

    return response[-50:].strip()




def run_batch_inference(model, processor, img_paths, questions, steps, decoding_strategy):
    """
    按 Qwen2.5-VL 官方 batch 模板写法
    """
    batch_messages = [create_messages(p, q) for p, q in zip(img_paths, questions)]

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]

    image_inputs, video_inputs = process_vision_info(batch_messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            slvr_steps=[steps] * len(img_paths),
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )


    return output_texts


def is_cuda_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


# ==========================================
# <<< changed >>> 遍历目录下所有 json（递归）
# ==========================================
def iter_all_json_files(input_dir: str):
    input_dir = Path(input_dir)
    for p in sorted(input_dir.rglob("*.json")):
        yield p


def build_output_path(json_path: Path) -> Path:
    """保持输入目录结构，避免同名 json 结果互相覆盖。"""
    rel = json_path.relative_to(Path(Config.INPUT_DIR))
    out_path = Path(Config.OUTPUT_SUBDIR) / rel.parent / f"{rel.stem}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def build_rank_tmp_path(json_path: Path, rank: int) -> Path:
    """分布式下每个 rank 的临时输出路径。"""
    rel = json_path.relative_to(Path(Config.INPUT_DIR))
    tmp_path = Path(Config.OUTPUT_SUBDIR) / ".dist_tmp" / rel.parent / f"{rel.stem}.rank{rank}.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    return tmp_path


# ==========================================
# <<< changed >>> 单个json文件推理并保存一个结果json
# ==========================================
def infer_one_json_file(json_path: Path, model, processor, rank: int = 0, world_size: int = 1):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[Skip] {json_path} is not a list json.")
        return None

    # 多卡模式下：所有 rank 共同处理同一个 json，每个 rank 处理切片样本。
    if world_size > 1:
        indexed_items = [(i, item) for i, item in enumerate(data) if i % world_size == rank]
    else:
        indexed_items = list(enumerate(data))

    results = []
    base_bs = Config.BATCH_SIZE

    for start in tqdm(range(0, len(indexed_items), base_bs), desc=f"推理: {json_path.name} [R{rank}]", leave=False):
        batch = indexed_items[start:start + base_bs]
        valid_samples = []

        for orig_idx, item in batch:
            item_id = item.get('id', 'unknown')
            image_paths = item.get('images', [])
            if not image_paths:
                continue

            current_img_path = image_paths if len(image_paths) > 1 else image_paths[0]

            conversations = item.get('conversations', [])
            if not conversations:
                continue

            raw_question = conversations[0]['value'].replace('<image>', '').strip()
            input_prompt = raw_question + Config.TASK_INSTRUCTION

            valid_samples.append({
                "__orig_idx": orig_idx,
                "id": item_id,
                "image": current_img_path,
                "question": raw_question,
                "full_question": input_prompt,
                "ground_truth": conversations[1]['value'] if len(conversations) > 1 else None,
            })

        if len(valid_samples) == 0:
            continue

        pos = 0
        dynamic_bs = min(base_bs, len(valid_samples))
        while pos < len(valid_samples):
            cur_bs = min(dynamic_bs, len(valid_samples) - pos)
            cur = valid_samples[pos:pos + cur_bs]
            cur_imgs = [x["image"] for x in cur]
            cur_questions = [x["full_question"] for x in cur]

            try:
                batch_responses = run_batch_inference(
                    model=model,
                    processor=processor,
                    img_paths=cur_imgs,
                    questions=cur_questions,
                    steps=Config.STEPS,
                    decoding_strategy=Config.DECODING_STRATEGY
                )
                for meta, resp in zip(cur, batch_responses):
                    results.append({
                        "__orig_idx": meta["__orig_idx"],
                        "id": meta["id"],
                        "image": meta["image"],
                        "question": meta["question"],
                        "task_instruction": Config.TASK_INSTRUCTION,
                        "full_response": resp,
                        "extracted_answer": extract_answer(resp),
                        "ground_truth": meta["ground_truth"],
                    })
                pos += cur_bs
                continue
            except Exception as e:
                if is_cuda_oom_error(e) and cur_bs > 1:
                    next_bs = max(1, cur_bs // 2)
                    dynamic_bs = next_bs
                    print(
                        f"[OOM][R{rank}] file={json_path.name} pos={pos} "
                        f"reduce batch {cur_bs} -> {next_bs}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                if is_cuda_oom_error(e):
                    print(f"[OOM][R{rank}] 单样本也OOM，写入错误占位: {json_path.name}")
                    meta = cur[0]
                    results.append({
                        "__orig_idx": meta["__orig_idx"],
                        "id": meta["id"],
                        "image": meta["image"],
                        "question": meta["question"],
                        "task_instruction": Config.TASK_INSTRUCTION,
                        "full_response": f"[ERROR] OOM: {str(e)}",
                        "extracted_answer": "",
                        "ground_truth": meta["ground_truth"],
                    })
                    pos += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                print(f"[Batch Error] file={json_path} err={str(e)}，退化为逐条推理该chunk...")
                for meta in cur:
                    ip = meta["image"]
                    q = meta["full_question"]
                    resp = ""
                    extracted_answer = ""

                    try:
                        resp = run_single_inference(
                            model=model,
                            processor=processor,
                            img_path=ip,
                            question=q,
                            steps=Config.STEPS,
                            decoding_strategy=Config.DECODING_STRATEGY
                        )
                        extracted_answer = extract_answer(resp)
                    except Exception as e2:
                        resp = f"[ERROR] {str(e2)}"

                    results.append({
                        "__orig_idx": meta["__orig_idx"],
                        "id": meta["id"],
                        "image": meta["image"],
                        "question": meta["question"],
                        "task_instruction": Config.TASK_INSTRUCTION,
                        "full_response": resp,
                        "extracted_answer": extracted_answer,
                        "ground_truth": meta["ground_truth"],
                    })
                pos += cur_bs
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    return results


def main():
    init_distributed_if_needed()

    rank0_print("配置加载完成:")
    rank0_print(f"- 使用GPU(环境变量 GPU_IDS): {GPU_IDS if GPU_IDS else '未设置(使用全部可见GPU)'}")
    rank0_print(f"- 模型路径: {Config.MODEL_PATH}")
    rank0_print(f"- 输入目录: {Config.INPUT_DIR}")
    rank0_print(f"- 输出目录: {Config.OUTPUT_SUBDIR}")
    rank0_print(f"- batch_size: {Config.BATCH_SIZE}")
    rank0_print(f"- 分布式: {IS_DISTRIBUTED} | world_size={WORLD_SIZE}")

    print(f"[Rank {RANK}] 正在加载模型到 {DEVICE} ...")
    model, processor = load_model_and_processor(Config.MODEL_PATH)
    print(f"[Rank {RANK}] 模型加载完成!")

    total_files = 0
    total_written_files = 0

    all_json_files = list(iter_all_json_files(Config.INPUT_DIR))
    print(f"[Rank {RANK}] 待处理 json 文件数: {len(all_json_files)}")

    for json_path in all_json_files:
        total_files += 1
        out_path = build_output_path(json_path)

        if IS_DISTRIBUTED:
            # 多卡共同处理一个 json：每个 rank 写本地切片，rank0 合并。
            tmp_path = build_rank_tmp_path(json_path, RANK)
            try:
                local_results = infer_one_json_file(
                    json_path,
                    model,
                    processor,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
                if local_results is None:
                    local_results = []
            except Exception as e:
                print(f"[Rank {RANK}] [Error] local shard failed for {json_path}: {e}")
                local_results = []

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(local_results, f, ensure_ascii=False, indent=2)

            distributed_barrier_if_needed()

            if RANK == 0:
                merged = []
                for r in range(WORLD_SIZE):
                    rp = build_rank_tmp_path(json_path, r)
                    if not rp.exists():
                        print(f"[Rank 0] [Warn] missing shard file: {rp}")
                        continue
                    try:
                        with open(rp, "r", encoding="utf-8") as f:
                            shard_data = json.load(f)
                        if isinstance(shard_data, list):
                            merged.extend(shard_data)
                    except Exception as e:
                        print(f"[Rank 0] [Warn] failed to load shard {rp}: {e}")

                merged.sort(key=lambda x: x.get("__orig_idx", 10**18))
                for item in merged:
                    item.pop("__orig_idx", None)

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)

                total_written_files += 1
                print(f"[Rank 0] [Done] {json_path} -> {out_path} (n={len(merged)})")

            distributed_barrier_if_needed()
        else:
            try:
                results = infer_one_json_file(json_path, model, processor)
                if results is None:
                    continue

                for item in results:
                    item.pop("__orig_idx", None)

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                total_written_files += 1
                print(f"[Rank {RANK}] [Done] {json_path} -> {out_path} (n={len(results)})")

            except Exception as e:
                print(f"[Rank {RANK}] [Error] processing {json_path}: {e}")
                continue

    if IS_DISTRIBUTED:
        if RANK == 0:
            tmp_root = Path(Config.OUTPUT_SUBDIR) / ".dist_tmp"
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            print("\n====== DONE (ALL RANKS) ======")
            print(f"Total json files scanned : {len(all_json_files)}")
            print(f"Total result files saved : {total_written_files}")
            print(f"Output dir              : {Config.OUTPUT_SUBDIR}")
    else:
        print("\n====== DONE ======")
        print(f"Total json files scanned : {total_files}")
        print(f"Total result files saved : {total_written_files}")
        print(f"Output dir              : {Config.OUTPUT_SUBDIR}")

    cleanup_distributed_if_needed()


if __name__ == "__main__":
    if not auto_launch_multi_gpu_if_needed():
        main()
