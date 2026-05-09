# SLVR Release

This repository is the public code release for SLVR and its training scripts.

## What is included

- Stage-1 SFT and Stage-2 GRPO training scripts.
- Model and trainer code used by the release.
- Dataset templates for SLV-Set and SV-QA.

The repository has been sanitized for public release. Machine-specific absolute paths, generated W&B run folders, and local experiment outputs are not part of the public tree.

## Environment

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

## Dataset templates

The file [meta_viscot.json](meta_viscot.json) is a template for Stage-1 training data. Replace the placeholder values with your own annotation JSONL file and image root before running locally.

## Training scripts

Stage-1 SFT:

```bash
bash scripts/finetune_lvr_stage1_7b_viscot.sh
```

Stage-2 GRPO:

```bash
bash scripts/finetune_lvr_stage2_7b_mgrpo_viscot.sh
```

The scripts are written to avoid hard-coded local paths. Provide your own `MODEL_NAME`, `DATA_PATH`, `IMAGE_FOLDER`, `CHKPT_PATH`, and `OUTPUT_DIR` through environment variables when needed.

## Notes

- `wandb/` is runtime output and should not be committed.
- Stage-2 judge settings are controlled by environment variables rather than embedded host-specific paths.

## License

Apache-2.0
