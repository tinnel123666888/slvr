#!/bin/bash
set -euo pipefail

SLVR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SLVR_ROOT" || exit 1
export PYTHONPATH="$SLVR_ROOT:$SLVR_ROOT/src:$PYTHONPATH"
# model configs
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
export WANDB_PROJECT="LVR-Qwen25-VL-7B-SFT-STAGE-1-450k-text_new1"
export WANDB_MODE=offline
export WANDB_DISABLE_GIT=true
# Data Config
DATA_PACKING=True
DS_SKIP_CUDA_CHECK=1
LST=4096
MAX_INSTANCE_PER_BATCH=4
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))


RANDOM_SEED=42
DATA_PATH="${DATA_PATH:-$SLVR_ROOT/meta_viscot.json}"

# General training params
GLOBAL_BATCH_SIZE=64       # global_batch_size becomes irrelevant when use data packing
BATCH_PER_DEVICE=1           # if use data packing, BS should always be 1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=8
#GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# LLM-related params
LR=1e-5
LVR_HEAD=False

# LVR-related params
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1
LAMBDA_LVR_text=0.1

MAX_TOKEN=5120
MIN_TOKEN=128


RUN_NAME="text_Stage1_${LVR_LOSS_FCT}LVRLossLambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
# ONLINE=True to enable online checkpointing with OCI
ONLINE=False
OUTPUT_DIR="${OUTPUT_DIR:-stage1_checkpoints/}"


# if continue training, set checkpoint_name = checkpoint to continue;
# --checkpoint_name checkpoint-4000


deepspeed src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT\
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --lvr_text_head True \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --loss_lvr_text_lambda $LAMBDA_LVR_text \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --online_checkpoint $ONLINE \
    --output_dir "$OUTPUT_DIR" \
    --max_steps -1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 50 \
    --dataloader_num_workers 8 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
    --enable_data_packing True \
    # save_total_limit is for local storage only, no limit for online checkpointing