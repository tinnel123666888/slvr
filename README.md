# SLVR ICML2026

This repository contains the public training and inference code for SLVR (ICML 2026 submission version).

## Environment

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

## 1) 数据集怎么准备

### Stage-1 SFT 数据

Stage-1 读取 `meta_viscot.json`，格式如下：

```json
[
  {
    "ds_name": "viscot",
    "data_path": "/path/to/your/training_data.jsonl",
    "image_folder": "/path/to/your/images/",
    "ds_type": "Q_A"
  }
]
```

你需要填写两条路径：

- `data_path`: 训练标注 JSONL 路径。
- `image_folder`: 该 JSONL 里图片路径对应的根目录。

### Stage-2 M-GRPO 数据

Stage-2 通过环境变量读取：

- `DATA_PATH`: Stage-2 训练 JSON（例如 2q 数据）。
- `IMAGE_FOLDER`: 图像根目录。
- `CHKPT_PATH`: Stage-1 checkpoint 路径。

## 2) Judge 怎么配置（Stage-2）

Stage-2 judge 不再写死路径，全部用环境变量：

- `MGRPO_JUDGE_BROKER_URL`: judge broker 服务地址。
- `MGRPO_JUDGE_BROKER_APPID`: broker appid / 模型路由标识。
- `MGRPO_JUDGE_PORT`: judge 推理端口（默认 8000）。
- `MGRPO_JUDGE_WORKERS`: judge 并发线程数。
- `MGRPO_JUDGE_TIMEOUT`: 单请求超时秒数。
- `MGRPO_JUDGE_IP_REFRESH_INTERVAL`: IP 刷新间隔。
- `MGRPO_DISABLE_LLM_JUDGE`: 设为 `1` 时关闭远程 judge，走 fallback。

最小示例：

```bash
export MGRPO_JUDGE_BROKER_URL="http://your-broker-host:port/api"
export MGRPO_JUDGE_BROKER_APPID="your-judge-appid"
export MGRPO_DISABLE_LLM_JUDGE=0
```

## 3) 训练怎么跑

### Stage-1

```bash
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="$(pwd)/meta_viscot.json"
export OUTPUT_DIR="stage1_checkpoints"
bash scripts/finetune_lvr_stage1_7b_viscot.sh
```

### Stage-2

```bash
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export CHKPT_PATH="/path/to/stage1_checkpoint"
export DATA_PATH="/path/to/viscot_363k_2q_qwen.json"
export IMAGE_FOLDER="/path/to/cot_images/"
export OUTPUT_DIR="stage2_mgrpo_checkpoints"

# judge (按需开启)
export MGRPO_JUDGE_BROKER_URL="http://your-broker-host:port/api"
export MGRPO_JUDGE_BROKER_APPID="your-judge-appid"

bash scripts/finetune_lvr_stage2_7b_mgrpo_viscot.sh
```

## 4) 推理怎么用

推理脚本：`inf_batch_dir_old.py`

你需要在脚本内 `Config` 里设置：

- `MODEL_PATH`: checkpoint 路径。
- `INPUT_DIR`: 待推理 JSON 文件目录（递归读取 `*.json`）。
- `BATCH_SIZE`: 可通过环境变量覆盖。

运行方式：

```bash
# 单卡或自动多卡
python inf_batch_dir_old.py

# 可选：指定可见 GPU 和 batch size
GPU_IDS=0,1 BATCH_SIZE=64 python inf_batch_dir_old.py
```

输出位置：

- 根目录：`./old_test/<checkpoint_parent>/test_slvr_results_<STEP>/`
- 每个输入 json 会生成对应的 `<name>_results.json`

## Notes

- `wandb/` 是运行产物，不应提交。
- 脚本内不再包含机器私有绝对路径。

## License

Apache-2.0
