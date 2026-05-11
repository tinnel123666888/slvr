#!/bin/bash
# ====================================================================
# M-GRPO Stage 2: Multi-query Group Relative Policy Optimization
# Dataset: your_training_dataset.json  (paired queries per region)
# Base model: Stage-1 checkpoint
# ====================================================================
set -euo pipefail

SSLVR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SSLVR_ROOT" || exit 1
export PYTHONPATH="$SSLVR_ROOT:$SSLVR_ROOT/src:$PYTHONPATH"

# ---- Model configs ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
CHKPT_PATH="${CHKPT_PATH:-/path/to/stage1_checkpoint}"

export WANDB_PROJECT="SLVR-Qwen25-VL-7B-MGRPO-STAGE2"
export WANDB_MODE=offline
export WANDB_DISABLE_GIT=true
export MGRPO_DISABLE_LLM_JUDGE=0
export MGRPO_JUDGE_BROKER_APPID="${MGRPO_JUDGE_BROKER_APPID:-}"
export MGRPO_JUDGE_BROKER_URL="${MGRPO_JUDGE_BROKER_URL:-}"
export MGRPO_JUDGE_PORT=8000
export MGRPO_JUDGE_WORKERS=16
export MGRPO_JUDGE_TIMEOUT=30
export MGRPO_JUDGE_IP_REFRESH_INTERVAL=5
export MGRPO_ACC_Q2_WEIGHT=${MGRPO_ACC_Q2_WEIGHT:-0.3}
export MGRPO_DISABLE_METRIC_GATHER=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=200000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# Increase NCCL heartbeat/blocking timeout (default 600s is too short for large models)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# Reduce fragmentation (suggested by PyTorch OOM message)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Data configs ----
DATA_PATH="${DATA_PATH:-/path/to/viscot_363k_2q_qwen.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/cot_images/}"
OUTPUT_DIR="${OUTPUT_DIR:-stage2_mgrpo_checkpoints/}"

# ---- Training configs ----
FREEZE_VISION=True
FREEZE_MERGER=True
DECODING_STRATEGY="latent"
SLVR_STEPS=8
SLVR_END_THRESHOLD=0.02
SLVR_END_CRITERION="mse"
LATENT_END_TOKEN=True
LR=5e-6
TEMP=0.9

NUM_DEVICES=8
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=4
NUM_GENERATIONS=2

MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=10024

# ---- M-GRPO reward weights ----
# format_reward now returns [0.0, 1.0] (malformed=0, valid=1).
# Keep format as an auxiliary signal (default 0.2) while accuracy remains primary.
LAMBDA_CONSISTENCY=0.3
LAMBDA_STABILITY=0.1
REWARD_WEIGHT_ACCURACY=1.0
REWARD_WEIGHT_FORMAT=${REWARD_WEIGHT_FORMAT:-0.2}
LAMBDA_SEM=0.01
LAMBDA_VIS=0.01

# ---- Dataset sampling ----
# Set to a fraction (e.g. 0.15) to use only that % of training data.
# Set to 1.0 to use full dataset.
DATASET_SAMPLE_RATIO=${DATASET_SAMPLE_RATIO:-0.15}
TAU_SEM=15.0
TAU_VIS=15.0
BETA=0.04

MIN_TOKEN=128
MAX_TOKEN=5120

RUN_NAME="MGRPO_Stage2_7B_decodingBy${DECODING_STRATEGY}_slvr${SLVR_STEPS}_LR${LR}_TEMP${TEMP}_lcons${LAMBDA_CONSISTENCY}_lstab${LAMBDA_STABILITY}"

# ---- Preflight inference (before training) ----
# Run a small pure-inference sanity check on random training samples.
# Set MGRPO_PREFLIGHT=0 to skip.
MGRPO_PREFLIGHT=${MGRPO_PREFLIGHT:-1}
MGRPO_PREFLIGHT_SAMPLES=${MGRPO_PREFLIGHT_SAMPLES:-5}

if [ "$MGRPO_PREFLIGHT" = "1" ]; then
    echo "[preflight] Running pure inference sanity check before training..."
    python src/train/preflight_mgrpo_infer.py \
        --model_pth "$CHKPT_PATH" \
        --data_path "$DATA_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$MGRPO_PREFLIGHT_SAMPLES" \
        --max_new_tokens "$MAX_COMPLETION_LENGTH" \
        --decoding_strategy "$DECODING_STRATEGY" \
        --slvr_steps "$SLVR_STEPS" \
        --criterion "$SLVR_END_CRITERION" \
        --slvr_end_threshold "$SLVR_END_THRESHOLD" \
        --latent_end_token "$LATENT_END_TOKEN"

    PRECHECK_PATH="$OUTPUT_DIR/preflight_inference.json"
    if [ -f "$PRECHECK_PATH" ]; then
        echo "[preflight] Done. Saved to: $PRECHECK_PATH"
    else
        echo "[preflight] ERROR: preflight output not found at $PRECHECK_PATH"
        exit 1
    fi
fi

deepspeed src/train/train_mgrpo.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2_optim_offload.json \
    --online_checkpoint False \
    --checkpoint_name "$CHKPT_PATH" \
    --model_id "$MODEL_NAME" \
    --latent_end_token $LATENT_END_TOKEN \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --dataset_sample_ratio $DATASET_SAMPLE_RATIO \
    --freeze_vision_tower $FREEZE_VISION \
    --freeze_merger $FREEZE_MERGER \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --temperature $TEMP \
    --beta $BETA \
    --num_train_epochs 1 \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --learning_rate $LR \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --ddp_timeout 3600 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --decoding_strategy $DECODING_STRATEGY \
    --slvr_steps $SLVR_STEPS \
    --criterion $SLVR_END_CRITERION \
    --slvr_end_threshold $SLVR_END_THRESHOLD \
    --lambda_consistency $LAMBDA_CONSISTENCY \
    --lambda_stability $LAMBDA_STABILITY \
    --lambda_sem $LAMBDA_SEM \
    --lambda_vis $LAMBDA_VIS \
    --tau_sem $TAU_SEM \
    --tau_vis $TAU_VIS \
    --reward_weights $REWARD_WEIGHT_ACCURACY $REWARD_WEIGHT_FORMAT
