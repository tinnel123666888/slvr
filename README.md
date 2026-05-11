# SSLVR: Semantic-Enriched Latent Visual Reasoning (ICML 2026)

Official training and inference code for **SSLVR**, accepted at ICML 2026.

---

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **SLV-Set** | Training data for both Stage-1 SFT and Stage-2 M-GRPO | [tinnel123/slv-set](https://huggingface.co/datasets/tinnel123/slv-set) |
| **SV-QA** | Evaluation benchmark | [tinnel123/sv-qa](https://huggingface.co/datasets/tinnel123/sv-qa) |

---

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate sslvr
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

---

## 1. Data Preparation

### Stage-1 SFT Data (SLV-Set)

Stage-1 reads from `meta_viscot.json`. Edit it to point to your local copy of SLV-Set:

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

- `data_path`: path to the training annotation JSONL file.
- `image_folder`: root directory for images referenced in the JSONL.

Download SLV-Set from [tinnel123/slv-set](https://huggingface.co/datasets/tinnel123/slv-set) and update these two paths accordingly.

### Stage-2 M-GRPO Data (SLV-Set)

Stage-2 also uses SLV-Set (the 2-question split). Set the following environment variables before running the script:

- `DATA_PATH`: path to the Stage-2 training JSON from SLV-Set.
- `IMAGE_FOLDER`: root directory for images.
- `CHKPT_PATH`: path to the Stage-1 checkpoint.

### Evaluation Benchmark (SV-QA)

SV-QA is our evaluation benchmark for measuring semantic visual reasoning. Download it from [tinnel123/sv-qa](https://huggingface.co/datasets/tinnel123/sv-qa).

---

## 2. Judge Configuration (Stage-2)

Stage-2 uses an LLM judge served via vLLM. Judge settings are configured through environment variables — **no code changes needed** for most setups.

The judge logic lives in `src/train/mgrpo_reward_funcs.py` (around lines 31–36). If you need to customize how the judge broker is called or add authentication, edit that file directly.

| Variable | Description |
|----------|-------------|
| `MGRPO_JUDGE_BROKER_URL` | URL of the judge broker service |
| `MGRPO_JUDGE_BROKER_APPID` | App ID / model routing key |
| `MGRPO_JUDGE_PORT` | vLLM inference port (default: `8000`) |
| `MGRPO_JUDGE_WORKERS` | Number of concurrent judge threads |
| `MGRPO_JUDGE_TIMEOUT` | Per-request timeout in seconds |
| `MGRPO_JUDGE_IP_REFRESH_INTERVAL` | Judge IP refresh interval |
| `MGRPO_DISABLE_LLM_JUDGE` | Set to `1` to disable remote judge (uses fallback) |

Minimal example:

```bash
export MGRPO_JUDGE_BROKER_URL="http://your-broker-host:port/api"
export MGRPO_JUDGE_BROKER_APPID="your-judge-appid"
export MGRPO_DISABLE_LLM_JUDGE=0
```

To run **without** a judge (e.g., for debugging):

```bash
export MGRPO_DISABLE_LLM_JUDGE=1
```

---

## 3. Training

### Stage-1: Supervised Fine-Tuning

```bash
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="$(pwd)/meta_viscot.json"
export OUTPUT_DIR="stage1_checkpoints"

bash scripts/finetune_slvr_stage1_7b_viscot.sh
```

### Stage-2: M-GRPO

```bash
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export CHKPT_PATH="/path/to/stage1_checkpoint"
export DATA_PATH="/path/to/slv-set/stage2_training_data.json"
export IMAGE_FOLDER="/path/to/images/"
export OUTPUT_DIR="stage2_mgrpo_checkpoints"

# Judge settings
export MGRPO_JUDGE_BROKER_URL="http://your-broker-host:port/api"
export MGRPO_JUDGE_BROKER_APPID="your-judge-appid"

bash scripts/finetune_slvr_stage2_7b_mgrpo_viscot.sh
```

---

## 4. Inference

The inference script is `inf_batch_dir_old.py`. Edit the `Config` class at the top of the file:

| Field | Description |
|-------|-------------|
| `MODEL_PATH` | Path to the trained checkpoint |
| `INPUT_DIR` | Directory of input JSON files (scanned recursively for `*.json`) |
| `OUTPUT_DIR` | Directory to write results |
| `STEPS` | Number of latent reasoning steps (default: `8`) |
| `DECODING_STRATEGY` | Set to `"latent"` for SSLVR-style inference |
| `BATCH_SIZE` | Batch size (can also be set via environment variable) |

The script auto-detects available GPUs and falls back gracefully on OOM.

```bash
python inf_batch_dir_old.py
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{sslvr2026,
  title     = {Semantic-Enriched Latent Visual Reasoning},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
