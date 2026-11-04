#!/usr/bin/env bash
# Build, check and (optionally) publish one IRx model:
#   bash scripts/build_model.sh irx1|irx2|pro|mini [--publish]
#
#   irx1  IRx-1       Qwen3.5-2B base        -> irx-1(-GGUF),     Q4_K_M
#   irx2  IRx-2       Qwen3.5-4B base        -> irx-2(-GGUF),     Q4_K_M
#   pro   IRx-2 Pro   Gemma 4 E4B base       -> irx-2-pro(-GGUF), Q4_K_M
#   mini  IRx-mini    Gemma 3 1B base        -> irx-mini(-GGUF),  Q8_0
#
# Steps: chat template (with this week's news digest) -> training data -> LoRA
# fine-tune -> GGUF (merged into the full-precision base, quantized once with an
# importance matrix) -> scripts/check_gguf.py. If the final checkpoint fails the
# check, earlier checkpoints are tried newest-first -- by hand this is how IRx-2
# (checkpoint 600) and IRx-mini (checkpoint 1200) shipped. The MLX build is made
# from the checkpoint that passed. Nothing is published unless a build passes.
#
# All weights, caches and intermediates live under $IRX_HOME (the external drive);
# models/ and data/processed* in the repo are symlinks into it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
source .venv/bin/activate

MODEL="${1:-}"
PUBLISH="${2:-}"
QWEN_TEMPLATE_ARGS=""
NO_DEFAULT_SYSTEM=""
DIGEST="--digest-file data/news_digest.json"
case "$MODEL" in
  # MAX_SEQ: largest training window that fits this 18GB Mac, measured on the
  # longest examples (Qwen 2B: 768 -> 9.7GB; Qwen 4B: 512 -> 10.8GB, 640 -> OOM;
  # Gemma E4B: 2048 -> 10.8GB; Gemma 3 1B: 1536 -> 4.6GB).
  irx1)
    ID="irx-1"; IDENTITY="IRx-1"; PROMPT_STYLE="default"
    export BASE_MODEL="mlx-community/Qwen3.5-2B-4bit"
    export GGUF_BASE="mlx-community/Qwen3.5-2B-bf16"
    export MODEL_NAME="IRx-1"
    export CHAT_TEMPLATE="scripts/chat_template.jinja"
    TEMPLATE_FAMILY="qwen"
    export QWEN35_FIX=1
    export MAX_SEQ="${MAX_SEQ:-768}"
    ;;
  irx2)
    ID="irx-2"; IDENTITY="IRx-2"; PROMPT_STYLE="default"
    export BASE_MODEL="mlx-community/Qwen3.5-4B-4bit"
    export GGUF_BASE="mlx-community/Qwen3.5-4B-bf16"
    export MODEL_NAME="IRx-2"
    export CHAT_TEMPLATE="scripts/chat_template_irx2.jinja"
    TEMPLATE_FAMILY="qwen"
    export QWEN35_FIX=1
    export MAX_SEQ="${MAX_SEQ:-512}"
    ;;
  pro)
    ID="irx-2-pro"; IDENTITY="IRx-2"; PROMPT_STYLE="clear"
    export BASE_MODEL="mlx-community/gemma-4-e4b-it-4bit"
    export GGUF_BASE="mlx-community/gemma-4-e4b-it-bf16"
    export MODEL_NAME="IRx-2 Pro"
    export CHAT_TEMPLATE="scripts/chat_template_irx2_pro.jinja"
    TEMPLATE_FAMILY="gemma"
    export QWEN35_FIX=0
    export MAX_SEQ="${MAX_SEQ:-2048}"
    # The f16 GGUF is ~15GB (Gemma 4 E-models' per-layer embeddings) -- too big
    # for this Mac's GPU during llama-imatrix, so that step runs on CPU.
    export IMATRIX_NGL=0
    ;;
  mini)
    # ~1B for 4GB-RAM phones. The 0.8B Qwen build looped on long answers (trip
    # itineraries) at PocketPal's default repeat penalty of 1.0; this base did not.
    ID="irx-mini"; IDENTITY="IRx-mini"; PROMPT_STYLE="clear"
    # Trained on the full-precision base: adapters learned on the 4-bit 1B and merged
    # into bf16 degraded more with every checkpoint (loops, leaked turn markers).
    export BASE_MODEL="mlx-community/gemma-3-1b-it-bf16"
    export GGUF_BASE="mlx-community/gemma-3-1b-it-bf16"
    export MODEL_NAME="IRx-mini"
    export CHAT_TEMPLATE="scripts/chat_template_irx_mini.jinja"
    TEMPLATE_FAMILY="gemma3"
    export QWEN35_FIX=0
    export MAX_SEQ="${MAX_SEQ:-1536}"
    export QUANT_TYPE="Q8_0"
    export NUM_LAYERS="${NUM_LAYERS:-8}"
    MLX_QUANT_BITS=8
    # Gemma 3 has no system role: an injected prompt lands inside the user's message,
    # and at 1B the model echoed it ("You are IRx-mini...") and listed the news digest
    # instead of answering. Identity is trained in; no prompt is injected or trained on.
    SYSTEM_PROMPT_FRACTION=0
    NO_DEFAULT_SYSTEM="--no-default-system"; DIGEST=""
    # With no identity prompt, 69 identity examples in ~1,700 weren't enough to stick.
    IDENTITY_REPEAT=3
    ;;
  *) echo "usage: $0 irx1|irx2|pro|mini [--publish]" >&2; exit 1 ;;
esac

export DATA_DIR="data/processed_${ID}"
export ADAPTER_DIR="models/${ID}-adapters"
export FP16_DIR="models/${ID}-merged-fp16"
export GGUF_F16="models/${ID}-f16.gguf"
export GGUF_QUANT="models/${ID}-${QUANT_TYPE:-Q4_K_M}.gguf"
MERGED_DIR="models/${ID}-merged"
snapshot() { python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$1"; }

echo "===== building $ID from $BASE_MODEL (max seq $MAX_SEQ) $(date '+%F %T') ====="

# Chat template from the base's stock one, with this week's news digest.
STOCK="$(mktemp)"
python - "$(snapshot "$BASE_MODEL")" "$STOCK" <<'PY'
import json, os, sys
d, out = sys.argv[1], sys.argv[2]
src = open(f"{d}/chat_template.jinja").read() if os.path.exists(f"{d}/chat_template.jinja") \
    else json.load(open(f"{d}/tokenizer_config.json"))["chat_template"]
open(out, "w").write(src)
PY
python scripts/make_chat_template.py --family "$TEMPLATE_FAMILY" --name "$IDENTITY" --base "$STOCK" \
  --out "$CHAT_TEMPLATE" --prompt-style "$PROMPT_STYLE" $DIGEST $NO_DEFAULT_SYSTEM
rm -f "$STOCK"

# Training data: shared sources + this model's identity examples.
mkdir -p "$DATA_DIR"
for f in data/processed/personal.jsonl data/processed/distilled_*.jsonl; do
  cp "$f" "$DATA_DIR/$(basename "$f")"
done
python scripts/generate_identity_data.py --name "$IDENTITY" --out "$DATA_DIR/identity.jsonl"
if [ "${IDENTITY_REPEAT:-1}" -gt 1 ]; then
  for _ in $(seq 2 "$IDENTITY_REPEAT"); do cat "$DATA_DIR/identity.jsonl"; done > "$DATA_DIR/identity.extra"
  cat "$DATA_DIR/identity.extra" >> "$DATA_DIR/identity.jsonl" && rm "$DATA_DIR/identity.extra"
fi
python scripts/combine_data.py --data-dir "$DATA_DIR" --base-model "$BASE_MODEL" --max-tokens "$MAX_SEQ" \
  --name "$IDENTITY" --prompt-style "$PROMPT_STYLE" --system-prompt-fraction "${SYSTEM_PROMPT_FRACTION:-0.6}"

# One pass over the data (batch size 1); checkpoints every 200 steps for the fallback below.
EPOCHS="${EPOCHS:-1}"
export ITERS="$(( $(wc -l < "$DATA_DIR/train.jsonl" | tr -d ' ') * EPOCHS ))"
rm -rf "$ADAPTER_DIR"
bash scripts/finetune.sh

# GGUF from an adapter set, then the behaviour check. Returns non-zero on failure.
build_and_check() {
  local adapters="$1"
  ADAPTER_DIR="$adapters" bash scripts/merge_and_quantize.sh
  rm -rf "$FP16_DIR" "$GGUF_F16"
  python scripts/check_gguf.py "$GGUF_QUANT" --name "$IDENTITY" --ngl "${CHECK_NGL:-99}"
}

PASSED=""
if build_and_check "$ADAPTER_DIR"; then
  PASSED="$ADAPTER_DIR"
else
  for ck in $(ls "$ADAPTER_DIR" | grep -E '^[0-9]+_adapters\.safetensors$' | sort -r); do
    echo "----- final build failed the check; trying checkpoint ${ck%%_*} -----"
    CK_DIR="models/${ID}-ck"; rm -rf "$CK_DIR"; mkdir -p "$CK_DIR"
    cp "$ADAPTER_DIR/adapter_config.json" "$CK_DIR/"
    cp "$ADAPTER_DIR/$ck" "$CK_DIR/adapters.safetensors"
    if build_and_check "$CK_DIR"; then PASSED="$CK_DIR"; break; fi
  done
fi
if [ -z "$PASSED" ]; then
  echo "===== $ID: no checkpoint passed the check -- not publishing =====" >&2
  exit 1
fi
echo "===== $ID: passed with $PASSED ====="

# MLX build from the adapters that passed.
rm -rf "$MERGED_DIR"
if [ -n "${MLX_QUANT_BITS:-}" ]; then
  # Trained on a full-precision base: fuse there, then quantize once.
  TMP_BF16="models/${ID}-mlx-bf16"; rm -rf "$TMP_BF16"
  mlx_lm.fuse --model "$(snapshot "$BASE_MODEL")" --adapter-path "$PASSED" --save-path "$TMP_BF16"
  mlx_lm.convert --hf-path "$TMP_BF16" --mlx-path "$MERGED_DIR" -q --q-bits "$MLX_QUANT_BITS"
  rm -rf "$TMP_BF16"
  python scripts/fix_mlx_gemma3.py "$MERGED_DIR" "$(snapshot "$BASE_MODEL")"
else
  mlx_lm.fuse --model "$(snapshot "$BASE_MODEL")" --adapter-path "$PASSED" --save-path "$MERGED_DIR"
fi
cp "$CHAT_TEMPLATE" "$MERGED_DIR/chat_template.jinja"
rm -f "$MERGED_DIR/README.md" "$MERGED_DIR/generation_config.json"
rm -rf "models/${ID}-ck"

echo "===== $ID ready: $MERGED_DIR $GGUF_QUANT $(date '+%F %T') ====="

if [ "$PUBLISH" = "--publish" ]; then
  export HF_HUB_OFFLINE=0
  python scripts/make_model_cards.py
  python scripts/publish_to_hf.py \
    --model-repo "ikppramesh/$ID" --gguf-repo "ikppramesh/$ID-GGUF" \
    --gguf-prefix "$ID" --gguf-quant "${QUANT_TYPE:-Q4_K_M}" \
    --merged-dir "$MERGED_DIR" --gguf-file "$GGUF_QUANT" --gguf-q4-0-file "models/none.gguf" \
    --gguf-readme "models/cards/${ID}-GGUF.md" \
    --commit-message "Weekly update $(date -u +%F): news up to $(python -c 'import json; print(json.load(open("data/news_digest.json"))["cutoff"])')"
fi
