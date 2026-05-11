"""
Reward functions for M-GRPO (Multi-query Group Relative Policy Optimization).

Two external reward components (loaded by load_reward_funcs via name_pred `_reward`):
  1. accuracy_reward  — LLM-as-judge correctness, with string-match fallback
  2. format_reward    — check <|vision_start|>...<|vision_end|> <sem>...</sem> <answer>...</answer>

Consistency and stability rewards are computed inside the trainer from hidden states.

LLM Judge architecture (mirrors judge_llm.py):
  - Background thread refreshes IP list from broker every 30 s
  - ThreadPoolExecutor (JUDGE_WORKERS threads) submits one call per (question, answer)
  - Falls back to exact/substring match when all LLM calls fail
"""

import os
import re
import time
import json
import socket
import struct
import random
import threading
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import requests

# ─── Configuration ────────────────────────────────────────────────────────────
BROKER_URL = os.getenv("MGRPO_JUDGE_BROKER_URL", "")  # set via MGRPO_JUDGE_BROKER_URL env var
BROKER_APPID = os.getenv("MGRPO_JUDGE_BROKER_APPID", "")  # set via MGRPO_JUDGE_BROKER_APPID env var
JUDGE_LLM_PORT = int(os.getenv("MGRPO_JUDGE_PORT", "8000"))
JUDGE_WORKERS = int(os.getenv("MGRPO_JUDGE_WORKERS", "16"))
JUDGE_TIMEOUT = float(os.getenv("MGRPO_JUDGE_TIMEOUT", "30.0"))
IP_REFRESH_INTERVAL = int(os.getenv("MGRPO_JUDGE_IP_REFRESH_INTERVAL", "5"))

# ─── Module-level shared IP pool ─────────────────────────────────────────────
_ip_pool: list[str] = []
_ip_pool_lock = threading.Lock()
_ip_refresh_thread: threading.Thread | None = None
_init_lock = threading.Lock()


def _is_judge_log_rank() -> bool:
    rank = os.getenv("RANK")
    local_rank = os.getenv("LOCAL_RANK")
    if rank is not None:
        return rank == "0"
    if local_rank is not None:
        return local_rank == "0"
    return True


def _judge_log(message: str, *, force: bool = False) -> None:
    verbose = os.getenv("MGRPO_JUDGE_VERBOSE", "0") == "1"
    if force or verbose:
        if _is_judge_log_rank():
            print(message, flush=True)


def _verify_ip(ip: str, port: int = JUDGE_LLM_PORT) -> bool:
    try:
        r = requests.get(f"http://{ip}:{port}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def _refresh_ip_pool():
    """Runs in a daemon thread: keeps _ip_pool updated from the broker."""
    global _ip_pool
    backoff = 5
    last_logged_pool: tuple[str, ...] = ()
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json", "User-Agent": "mgrpo-judge/1.0"})
    while True:
        try:
            res = sess.post(BROKER_URL, json={"appid": BROKER_APPID}, timeout=10)
            if res.status_code != 200:
                time.sleep(backoff); backoff = min(backoff * 2, 60); continue
            data = res.json()
            raw_list = data.get("machine_list", {}).get("list", [])
            ips = []
            for x in raw_list:
                try:
                    ip_int = x["attr"]["ip"]
                    ips.append(socket.inet_ntoa(struct.pack("!I", ip_int)))
                except Exception:
                    continue
            # Verify in parallel
            verified = []
            with ThreadPoolExecutor(max_workers=20) as vpool:
                results = list(vpool.map(_verify_ip, ips))
            for ip, ok in zip(ips, results):
                if ok:
                    verified.append(f"{ip}:{JUDGE_LLM_PORT}")
            with _ip_pool_lock:
                _ip_pool[:] = verified
            verified_tuple = tuple(verified)
            if verified_tuple != last_logged_pool:
                _judge_log(f"[MGRPO-Judge] IP pool refreshed: {len(verified)} IPs available", force=True)
                last_logged_pool = verified_tuple
            backoff = 5
        except Exception as e:
            _judge_log(f"[MGRPO-Judge] IP refresh error: {e}", force=True)
            backoff = min(backoff * 2, 60)
        time.sleep(IP_REFRESH_INTERVAL)


def _ensure_judge_ready():
    """Lazily start the refresh thread once."""
    global _ip_refresh_thread
    with _init_lock:
        if _ip_refresh_thread is None:
            t = threading.Thread(target=_refresh_ip_pool, daemon=True, name="mgrpo-ip-refresh")
            t.start()
            _ip_refresh_thread = t


def _pick_ip() -> str | None:
    with _ip_pool_lock:
        pool = list(_ip_pool)
    return random.choice(pool) if pool else None


def _llm_judge_single(ip_port: str, question: str, prediction: str, ground_truth: str):
    """Call vLLM judge for one (question, prediction, ground_truth) triple. Returns (bool|None, str)."""
    from openai import OpenAI

    question = question.strip() or "[Question not available]"
    prediction = prediction.strip() or "[No prediction provided]"
    ground_truth = ground_truth.strip() or "[No ground truth provided]"

    prompt = (
        "You are a strict evaluator.\n\n"
        "Given a question, its ground truth answer, and a model prediction.\n\n"
        f"===== QUESTION =====\n{question}\n\n"
        f"===== GROUND TRUTH ANSWER =====\n{ground_truth}\n\n"
        f"===== MODEL PREDICTION =====\n{prediction}\n\n"
        "===== TASK =====\n"
        "Decide whether the model prediction is semantically equivalent to the ground truth.\n"
        "- The prediction does NOT need to use the option letter.\n"
        "- If the meaning matches the correct option, it should be considered correct.\n"
        "- Compare the semantic meaning, not just surface form.\n\n"
        "Output ONLY:\n1  (if correct/equivalent)\n0  (if incorrect/not equivalent)\n"
    )
    try:
        client = OpenAI(api_key="EMPTY", base_url=f"http://{ip_port}/v1", timeout=JUDGE_TIMEOUT, max_retries=0)
        model_name = os.getenv("MGRPO_JUDGE_MODEL", "")
        if not model_name:
            model_name = client.models.list().data[0].id
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\b([01])\b", text)
        if m:
            return (m.group(1) == "1"), text
        tl = text.lower()
        positive = any(w in tl for w in ("correct", "right", "yes", "true", "1"))
        negative = any(w in tl for w in ("incorrect", "wrong", "no", "false", "0"))
        if positive and not negative:
            return True, text
        if negative and not positive:
            return False, text

        try:
            gt_norm = re.sub(r"\s+", " ", str(ground_truth or "").strip()).lower()
            pred_norm = re.sub(r"\s+", " ", str(prediction or "").strip()).lower()
            if gt_norm and gt_norm in pred_norm:
                return True, f"[FALLBACK_CONTAINS] {text}"
            gt_digits = re.sub(r"\D", "", gt_norm)
            pred_digits = re.sub(r"\D", "", pred_norm)
            if gt_digits and pred_digits and gt_digits == pred_digits:
                return True, f"[FALLBACK_DIGITS_EQ] {text}"
        except Exception:
            pass

        return None, text
    except Exception as e:
        return None, f"[EXCEPTION] {repr(e)}"


def _string_match_fallback(prediction: str, ground_truth: str) -> float:
    """Simple fallback when LLM judge is unavailable."""
    pred_m = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    pred = pred_m.group(1).strip() if pred_m else prediction.strip()
    gt_m = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    gt = gt_m.group(1).strip() if gt_m else ground_truth.strip()
    if not gt:
        return 0.0
    if pred.lower() == gt.lower():
        return 1.0
    if gt.lower() in pred.lower():
        return 0.5
    return 0.0


def _judge_one(question: str, prediction: str, ground_truth: str) -> float:
    """Single-item judge: try LLM, fall back to string match."""
    ip = _pick_ip()
    if ip:
        correct, _raw = _llm_judge_single(ip, question, prediction, ground_truth)
        if correct is True:
            return 1.0
        if correct is False:
            return 0.0
        # None means ambiguous — fall through
    return _string_match_fallback(prediction, ground_truth)


# ─── Public reward functions ──────────────────────────────────────────────────

def accuracy_reward(completions, assistant, prompts=None, **kwargs):
    """
    LLM-as-judge accuracy reward for M-GRPO.

    completions: list of dicts  [{"text1-answer": ..., "text2-answer": ...}, ...]
    assistant:   list of dicts  [{"text1-answer": ..., "text2-answer": ...}, ...]
    prompts:     list of prompt dicts (used to extract question text)

    Returns list[float] — one value per sample, averaged over both questions.
    """
    # Distributed training stability mode: bypass remote LLM judge and use deterministic fallback only.
    if os.getenv("MGRPO_DISABLE_LLM_JUDGE", "0") == "1":
        rewards = []
        for comp, sol in zip(completions, assistant):
            s1 = _string_match_fallback(comp.get("text1-answer", ""), sol.get("text1-answer", ""))
            s2 = _string_match_fallback(comp.get("text2-answer", ""), sol.get("text2-answer", ""))
            rewards.append((s1 + s2) / 2.0)
        return rewards

    _ensure_judge_ready()

    # Extract question text from prompts
    def _get_questions(prompt):
        q1, q2 = "", ""
        if not prompt:
            return q1, q2
        # Prompt is a list of role dicts; find user / content
        for msg in prompt:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text1":
                        q1 = item.get("text1", "")
                    elif item.get("type") == "text2":
                        q2 = item.get("text2", "")
        return q1, q2

    n = len(completions)
    prompts = prompts or [None] * n
    rewards = [0.0] * n

    # Submit all judge calls concurrently using a short-lived executor to avoid task backlog across steps.
    futures = {}
    per_item = {i: [] for i in range(n)}
    expected_per_item = {i: 0 for i in range(n)}
    judge_executor = ThreadPoolExecutor(max_workers=JUDGE_WORKERS, thread_name_prefix="mgrpo-judge-batch")
    try:
        for idx, (comp, sol, prompt) in enumerate(zip(completions, assistant, prompts)):
            q1, q2 = _get_questions(prompt.get("prompt", []) if isinstance(prompt, dict) else (prompt or []))
            for qi, key in enumerate(["text1-answer", "text2-answer"]):
                pred = comp.get(key, "")
                gt = sol.get(key, "")
                question = q1 if qi == 0 else q2
                fut = judge_executor.submit(_judge_one, question, pred, gt)
                futures[fut] = (idx, key, question, pred, gt)
                expected_per_item[idx] += 1

        batch_timeout = max(
            JUDGE_TIMEOUT + 5,
            int(math.ceil(max(len(futures), 1) / float(max(JUDGE_WORKERS, 1))) * (JUDGE_TIMEOUT + 2)),
        )
        done = set()
        try:
            for fut in as_completed(futures, timeout=batch_timeout):
                idx, key, question, pred, gt = futures[fut]
                done.add(fut)
                try:
                    score = fut.result(timeout=0)
                except Exception:
                    score = _string_match_fallback(pred, gt)
                per_item[idx].append(score)
        except FuturesTimeout:
            pass

        # Fill missing with fallback and try cancelling unfinished futures to reduce hanging background work.
        for fut, (idx, key, question, pred, gt) in futures.items():
            if fut not in done:
                fut.cancel()
                per_item[idx].append(_string_match_fallback(pred, gt))
    finally:
        judge_executor.shutdown(wait=False, cancel_futures=True)

    for idx in range(n):
        missing = expected_per_item[idx] - len(per_item[idx])
        if missing > 0:
            per_item[idx].extend([0.0] * missing)

    for idx, scores in per_item.items():
        rewards[idx] = sum(scores) / max(len(scores), 1)

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "/tmp/mgrpo_debug.log")
        ts = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a") as f:
            for idx in range(n):
                f.write(f"--- {ts} sample={idx} acc_reward={rewards[idx]:.3f} ---\n")
                f.write(f"comp: {completions[idx]}\ngt: {assistant[idx]}\n")

    return rewards


def format_reward(completions, **kwargs):
    """
    Check that each answer strictly follows the SLVR output format in order:
      1) Output starts with <|vision_start|>  (leading whitespace is stripped)
         Bypass (missing <|vision_start|>) is penalised: contributes -1.0 per sub-answer.
      2) At least MIN_SLVR_LATENT_STEPS <|vision_end|> tokens must appear between
         <|vision_start|> and <sem> (the visual latent placeholders).
      3) <sem> appears within N<=10 tokens after <|vision_start|>
         Tokens are counted by splitting on any special token (<|vision_end|>, spaces, etc.):
         each <|vision_end|> counts as 1 token; each non-empty text segment counts as 1 token.
      4) <answer> appears within 1-3 tokens after <sem>
         (same token counting: <|vision_end|> and non-empty text segments each count as 1)
      5) </answer> must appear after <answer>

        Per sub-answer scoring:
             0.0  full format correct (checks 1-5)
            -0.5  any malformed case (including bypass)

        Returns sum over both sub-answers: range [-1.0, 0.0].
    """
    SLVR_START = "<|vision_start|>"
    SLVR_TEXT_START = "<sem>"
    SLVR_END = "<|vision_end|>"
    ANSWER_START = "<answer>"
    ANSWER_END = "</answer>"
    MAX_SLVR_STEPS = 10
    MAX_ANSWER_GAP = 3
    MIN_SLVR_LATENT_STEPS = 2   # require at least 2 <|vision_end|> latent tokens in visual span

    def _count_tokens_in_segment(segment: str) -> int:
        """Count tokens in a text segment between special markers.

        Split by <|vision_end|>: each separator is 1 token, each non-empty
        text piece (stripped) between separators is 1 token.
        """
        parts = segment.split(SLVR_END)
        slvr_end_count = len(parts) - 1
        non_end_tokens = sum(1 for p in parts if p.strip())
        return slvr_end_count + non_end_tokens

    def check_format(text):
        """Return 0.0 (valid) or -0.5 (malformed)."""
        text = text.lstrip()

        # Must start with <|vision_start|>
        if not text.startswith(SLVR_START):
            return -0.5

        after_start = text[len(SLVR_START):]

        # <sem> must appear, and gap must be <=MAX_SLVR_STEPS tokens
        ts_idx = after_start.find(SLVR_TEXT_START)
        if ts_idx == -1:
            return -0.5
        between_start_ts = after_start[:ts_idx]
        if _count_tokens_in_segment(between_start_ts) > MAX_SLVR_STEPS:
            return -0.5

        # Require at least MIN_SLVR_LATENT_STEPS <|vision_end|> in visual span
        if between_start_ts.count(SLVR_END) < MIN_SLVR_LATENT_STEPS:
            return -0.5

        after_ts = after_start[ts_idx + len(SLVR_TEXT_START):]

        # <answer> must appear after <sem>, gap must be 1-3 tokens
        ans_idx = after_ts.find(ANSWER_START)
        if ans_idx == -1:
            return -0.5
        between_ts_ans = after_ts[:ans_idx]
        gap = _count_tokens_in_segment(between_ts_ans)
        if gap < 1 or gap > MAX_ANSWER_GAP:
            return -0.5

        # </answer> must appear after <answer>
        after_ans = after_ts[ans_idx + len(ANSWER_START):]
        if ANSWER_END not in after_ans:
            return -0.5

        return 0.0

    rewards = []
    for comp in completions:
        score = 0.0
        for key in ["text1-answer", "text2-answer"]:
            text = comp.get(key, "")
            score += check_format(text)
        rewards.append(score)   # range: -1.0 to 0.0
    return rewards
