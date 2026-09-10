#!/usr/bin/env bash
# QLoRA fine-tune Qwen3.5-2B (mlx-community's pre-quantized 4bit build) on the
# processed data. Requires: source .venv/bin/activate first (or run via `uv run`).
#
# --max-seq-length was previously 512 -- truncating longer training targets
# (personal-history conversations, and now the 600-900 token trip-planning
# examples) teaches the model to stop early, which is exactly the
# "responses cut off" symptom being fixed here. Raised to 1024, a
# conservative 2x rather than jumping further, since --num-layers 8 +
# --max-seq-length together previously caused a Metal command-buffer OOM
# crash (see prepare_data.py's MAX_HISTORY_MESSAGES/MAX_CHARS caps, added
# for the same incident). --num-layers is already reduced to 4 and
# --grad-checkpoint is on, both of which blunt memory growth from this --
# but this is the first real run at 1024, worth watching peak mem on.
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
  --iters 1450 \
  --max-seq-length 1024 \
  --steps-per-eval 100 \
  --steps-per-report 20 \
  --save-every 200 \
  --grad-checkpoint
