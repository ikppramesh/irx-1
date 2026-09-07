#!/usr/bin/env bash
# QLoRA fine-tune Qwen3.5-2B (mlx-community's pre-quantized 4bit build) on the
# processed data. Requires: source .venv/bin/activate first (or run via `uv run`).
set -euo pipefail

BASE_MODEL="mlx-community/Qwen3.5-2B-4bit"
DATA_DIR="data/processed"
ADAPTER_DIR="models/irx-1-adapters"

mlx_lm.lora \
  --model "$BASE_MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --batch-size 1 \
  --num-layers 4 \
  --iters 1300 \
  --max-seq-length 512 \
  --steps-per-eval 100 \
  --steps-per-report 20 \
  --save-every 200 \
  --grad-checkpoint
