#!/usr/bin/env bash
# QLoRA fine-tune Qwen3.5-2B (mlx-community's pre-quantized 4bit build) on the
# processed data. Requires: source .venv/bin/activate first (or run via `uv run`).
#
# --max-seq-length 768 (must match combine_data.py --max-tokens, which fits
# every example to this length instead of letting mlx_lm truncate it): 1024
# peaks above the ~13.4GB Metal limit on this 18GB Mac on the longest examples
# and crashed every scheduled retrain from 2026-09-11 to 09-29 with
# kIOGPUCommandBufferCallbackErrorOutOfMemory. Measured peaks on the longest
# examples: 768 -> 9.7GB, 896 -> 12.4GB, 1024 -> OOM. Memory is dominated by
# the full-model forward over the sequence, not the LoRA layers (--num-layers 2
# still OOMs at 1024), so 768 is the lever. It leaves headroom for other apps
# being open while the launchd job runs.
#
# --mask-prompt: loss only on the final assistant answer, not on user text.
#
# Memory cap (TRAIN_MEMORY_GB, default 11): the scheduled runs on 2026-10-29 and
# 2026-11-04 froze the whole Mac into a kernel panic ("watchdog timeout") minutes
# into training -- MLX wired most of the 18GB while other apps were open. The cap
# makes MLX free its cache and stay under the limit instead.
set -euo pipefail

# Defaults build IRx-1; scripts/build_irx2.sh overrides these for IRx-2 variants.
BASE_MODEL="${BASE_MODEL:-mlx-community/Qwen3.5-2B-4bit}"
DATA_DIR="${DATA_DIR:-data/processed}"
ADAPTER_DIR="${ADAPTER_DIR:-models/irx-1-adapters}"
MAX_SEQ="${MAX_SEQ:-768}"
NUM_LAYERS="${NUM_LAYERS:-4}"
ITERS="${ITERS:-1450}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

TRAIN_MEMORY_GB="${TRAIN_MEMORY_GB:-11}"

# Runs mlx_lm.lora with MLX capped at TRAIN_MEMORY_GB. Extra args are passed through.
lora() {
  python - "$TRAIN_MEMORY_GB" "$@" <<'PY'
import sys
import mlx.core as mx
gb = float(sys.argv.pop(1))
mx.set_memory_limit(int(gb * 2**30))
mx.set_wired_limit(int(gb * 0.7 * 2**30))
mx.set_cache_limit(int(2 * 2**30))
from mlx_lm.lora import main
sys.argv[0] = "mlx_lm.lora"
main()
PY
}

# Even capped, a GPU out-of-memory can still happen when other apps hold GPU memory
# (IRx-mini stopped at step 1260 of ~1900 on 2026-11-04 with a 6.5GB peak). Resume
# from the newest checkpoint instead of losing the run: up to 3 resumes. Checkpoints
# from a resumed segment are renamed to their true step numbers.
mkdir -p "$ADAPTER_DIR"
DONE=0
for attempt in 1 2 3 4; do
  SEG="$ADAPTER_DIR"
  RESUME=()
  if [ "$DONE" -gt 0 ]; then
    SEG="$ADAPTER_DIR/resume$attempt"; rm -rf "$SEG"
    RESUME=(--resume-adapter-file "$ADAPTER_DIR/$(printf '%07d' "$DONE")_adapters.safetensors")
    echo "----- resuming at step $DONE of $ITERS (attempt $attempt) -----"
  fi
  if lora --model "$BASE_MODEL" --train --data "$DATA_DIR" --adapter-path "$SEG" \
      --batch-size 1 --num-layers "$NUM_LAYERS" --iters "$((ITERS - DONE))" \
      --learning-rate "$LEARNING_RATE" --max-seq-length "$MAX_SEQ" --mask-prompt \
      --steps-per-eval 100 --steps-per-report 20 --save-every 200 --grad-checkpoint \
      "${RESUME[@]+"${RESUME[@]}"}"; then
    OK=1
  else
    OK=0
  fi
  if [ "$SEG" != "$ADAPTER_DIR" ]; then
    for f in "$SEG"/[0-9]*_adapters.safetensors; do
      [ -e "$f" ] || continue
      n=$((10#$(basename "$f" | cut -d_ -f1)))
      mv "$f" "$ADAPTER_DIR/$(printf '%07d' $((DONE + n)))_adapters.safetensors"
    done
    [ "$OK" = 1 ] && cp "$SEG/adapters.safetensors" "$ADAPTER_DIR/adapters.safetensors"
    rm -rf "$SEG"
  fi
  [ "$OK" = 1 ] && exit 0
  LAST=$(ls "$ADAPTER_DIR" | grep -E '^[0-9]+_adapters\.safetensors$' | sort | tail -1)
  [ -n "$LAST" ] || { echo "training failed before the first checkpoint" >&2; exit 1; }
  DONE=$((10#${LAST%%_*}))
done
echo "training failed after 3 resumes" >&2
exit 1
