#!/usr/bin/env bash
# Full recurring pipeline, run automatically after every news-fetch cron
# cycle (see scripts/com.irx1.retrain.plist, StartInterval matches
# scripts/com.irx1.newsfetch.plist): fetch news -> turn it into a small
# rolling training set -> LoRA fine-tune -> fuse (both MLX-quantized and
# GGUF paths) -> publish both Hugging Face repos.
#
# Known, tested limitation (see the model card's Limitations section):
# fine-tuning does not reliably teach the model new facts, news least of
# all -- that finding is exactly why scripts/news_context.py and
# src/utils/news.ts (retrieval, not retraining) exist. This script runs on
# a recurring schedule anyway because the project owner explicitly chose
# that after being told this tradeoff. The reliable current-events path is
# unaffected by anything here.
set -euo pipefail
cd "$(dirname "$0")/.."
# launchd runs this job with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin)
# that doesn't include Homebrew's bin dir, where llama-quantize lives (brew
# install llama.cpp). Every one of the first 8 automated cycles died at
# that exact step with "command not found" and never got to publish --
# only found because it worked fine when run interactively, where the
# shell's own PATH already has this.
export PATH="/opt/homebrew/bin:$PATH"
source .venv/bin/activate

BASE_MODEL="mlx-community/Qwen3.5-2B-4bit"
ADAPTER_DIR="models/irx-1-adapters"
MERGED_DIR="models/irx-1-merged"
FP16_DIR="models/irx-1-merged-fp16"
GGUF_F16="models/irx-1-qwen3.5-2b-f16.gguf"
GGUF_QUANT="models/irx-1-qwen3.5-2b-Q4_K_M.gguf"

echo "===== retrain run: $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

echo "[1/7] Fetching latest news (guarantees this cycle's data is fresh)"
python scripts/fetch_news.py --json-out docs/news.json

echo "[2/7] Turning recent articles into rolling training examples"
python scripts/generate_news_training_data.py

echo "[3/7] Recombining training data"
python scripts/combine_data.py

echo "[4/7] Fine-tuning (mlx_lm.lora)"
bash scripts/finetune.sh

echo "[5/7] Fusing adapters into quantized MLX weights"
# mlx_lm.fuse overwrites README.md in --save-path with its own minimal
# auto-generated stub (discovered the hard way: it clobbered the full model
# card and re-introduced a base_model/license_link naming Qwen, which is
# explicitly not wanted on this repo). Back up and restore the curated
# README + diagram assets around the fuse call.
cp "$MERGED_DIR/README.md" /tmp/irx1_readme_backup.md
mlx_lm.fuse \
  --model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_DIR" \
  --save-path "$MERGED_DIR"
cp /tmp/irx1_readme_backup.md "$MERGED_DIR/README.md"

echo "[6/7] Building GGUF (dequantized fuse -> bugfix -> convert -> quantize)"
bash scripts/merge_and_quantize.sh

echo "[7/7] Publishing to Hugging Face"
python scripts/publish_to_hf.py \
  --merged-dir "$MERGED_DIR" \
  --gguf-file "$GGUF_QUANT" \
  --commit-message "Automated retrain $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# fp16 intermediates are multi-GB and only needed mid-pipeline -- the earlier
# disk-full incident during manual GGUF debugging is exactly what this avoids.
rm -rf "$FP16_DIR" "$GGUF_F16"

# Commit the refreshed local news snapshot (the fetch-news.yml GitHub Action
# also publishes this independently; if both raced and pushed, this pull
# --rebase reconciles before this job's own push).
git add docs/news.json
if ! git diff --staged --quiet; then
  git pull --rebase --autostash || true
  git commit -m "Automated retrain: refresh news snapshot [skip ci]"
  git push || echo "git push failed -- will retry next cycle"
fi

echo "===== retrain run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
