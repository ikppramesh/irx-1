#!/usr/bin/env bash
# Merge LoRA adapters into the base model, then quantize to GGUF for on-device use
# (llama.cpp-based runtimes: PocketPal, LM Studio, llama.rn mobile apps, Ollama).
# Requires: source .venv/bin/activate, and llama.cpp cloned as a sibling dir:
#   git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
#   pip install -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
#     (in a SEPARATE venv, or expect it to downgrade transformers/numpy/tokenizers
#      here — it did, and broke mlx_lm's tokenizer loading, the first time this
#      was tried; restore this project's requirements.txt afterward if so)
set -euo pipefail

# Defaults build IRx-1; scripts/build_irx2.sh overrides these for IRx-2 variants.
# GGUF_BASE is the FULL-PRECISION build of the base the adapters were trained on.
# Merging into the dequantized 4-bit training base (the old approach) compounded
# two rounds of 4-bit error; in testing that build copied earlier answers
# verbatim in multi-turn chats and looped ("pizza, pizza, ...") while the same
# adapters merged into full precision did neither.
GGUF_BASE="${GGUF_BASE:-mlx-community/Qwen3.5-2B-bf16}"
ADAPTER_DIR="${ADAPTER_DIR:-models/irx-1-adapters}"
DATA_DIR="${DATA_DIR:-data/processed}"
MERGED_DIR="${FP16_DIR:-models/irx-1-merged-fp16}"
LLAMA_CPP_DIR="../llama.cpp"
GGUF_F16="${GGUF_F16:-models/irx-1-f16.gguf}"
GGUF_QUANT="${GGUF_QUANT:-models/irx-1-Q4_K_M.gguf}"
IMATRIX="${GGUF_F16%.gguf}-imatrix.gguf"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
MODEL_NAME="${MODEL_NAME:-IRx-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-scripts/chat_template.jinja}"
# The two MLX export bugs below are Qwen3.5-specific; 0 for other families.
QWEN35_FIX="${QWEN35_FIX:-1}"

# Fuse from the local snapshot path, not the repo id: given a repo id, mlx_lm
# copies that repo's README.md as the model card and crashes when the repo has
# none (mlx-community/Qwen3.5-4B-bf16 doesn't).
GGUF_BASE_PATH="$(python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$GGUF_BASE")"
mlx_lm.fuse \
  --model "$GGUF_BASE_PATH" \
  --adapter-path "$ADAPTER_DIR" \
  --save-path "$MERGED_DIR"

# mlx_lm.fuse doesn't copy these tokenizer files. Without tokenizer.model the Gemma 3
# converter falls back to tokenizer.json and fails on Gemma's vocab; without
# added_tokens.json / special_tokens_map.json, <start_of_turn> and <end_of_turn> are
# written as plain text tokens, so turns aren't parsed and the model writes fake turns.
for f in tokenizer.model added_tokens.json special_tokens_map.json; do
  if [ -f "$GGUF_BASE_PATH/$f" ] && [ ! -f "$MERGED_DIR/$f" ]; then
    cp "$GGUF_BASE_PATH/$f" "$MERGED_DIR/"
  fi
done
# The SentencePiece converter reads token types from tokenizer_config.json's
# added_tokens_decoder, which the re-saved config no longer has. Fine-tuning doesn't
# change the tokenizer, so restore the base's config -- minus its chat_template, which
# would override our chat_template.jinja.
if [ -f "$MERGED_DIR/tokenizer.model" ]; then
  python - "$GGUF_BASE_PATH/tokenizer_config.json" "$MERGED_DIR/tokenizer_config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
config.pop("chat_template", None)
json.dump(config, open(sys.argv[2], "w"), indent=2)
PY
fi

# mlx_lm.fuse saves a tied lm_head as its own tensor (Gemma 3 1B). The converter then
# writes a second copy of the 262k-token embedding table (+306MB at Q8_0). Drop it when
# it's identical to embed_tokens; llama.cpp ties them when output.weight is absent.
python - "$MERGED_DIR" <<'PY'
import glob, json, os, sys
import mlx.core as mx
for path in glob.glob(f"{sys.argv[1]}/model*.safetensors"):
    w = mx.load(path)
    emb = next((k for k in w if k.endswith("embed_tokens.weight")), None)
    if "lm_head.weight" in w and emb and mx.array_equal(w["lm_head.weight"], w[emb]).item():
        del w["lm_head.weight"]
        mx.eval(w)  # mx.load is lazy; read everything before replacing the file
        tmp = path.removesuffix(".safetensors") + ".tmp.safetensors"
        mx.save_safetensors(tmp, w, metadata={"format": "mlx"})
        os.replace(tmp, path)
        index = f"{sys.argv[1]}/model.safetensors.index.json"
        if os.path.exists(index):
            j = json.load(open(index))
            j["weight_map"].pop("lm_head.weight", None)
            json.dump(j, open(index, "w"), indent=2)
        print(f"Dropped tied lm_head.weight from {path}")
PY

# Two real bugs in mlx_lm.convert's Qwen3.5 export silently corrupt GGUF
# conversion — the file converts and loads without any error, but generates
# complete garbage. Found by diffing every tensor against the raw HF
# checkpoint; see the docstring in fix_gguf_mlx_conversion.py for the full
# story (conv1d.weight layout + a Gemma-style RMSNorm weight offset). The bf16
# mlx-community build carries the same two transforms as the 4-bit one.
if [ "$QWEN35_FIX" = "1" ]; then
  python scripts/fix_gguf_mlx_conversion.py "$MERGED_DIR"
fi

# mlx_lm.fuse also writes an auto-generated README.md into $MERGED_DIR with
# `base_model: ...` and a license link in its frontmatter. convert_hf_to_gguf.py
# reads exactly that file as this model's "model card" and copies those fields
# straight into the GGUF's general.base_model.* / general.license.link metadata.
# Deleting it here (a throwaway intermediate, not the published README) stops
# that heuristic from firing; --model-name sets the name we actually want.
rm -f "$MERGED_DIR/README.md"

# convert_hf_to_gguf.py embeds chat_template.jinja as tokenizer.chat_template,
# which is what PocketPal/LM Studio/llama.cpp apps render. Replace the stock one
# (thinking always off, default identity system prompt) -- see
# scripts/make_chat_template.py.
cp "$CHAT_TEMPLATE" "$MERGED_DIR/chat_template.jinja"

# Written into the GGUF as general.sampling.* -- the defaults llama.cpp-based
# apps use when the user hasn't set their own. The repeat penalty is the main
# guard against a small model looping on a phrase.
cp scripts/generation_config.json "$MERGED_DIR/generation_config.json"

# --no-mtp: this checkpoint doesn't carry Qwen3.5's optional multi-token-
# prediction head (not needed for normal, non-speculative-decoding inference).
# (Qwen only -- the Gemma 4 converter rejects the flag.)
MTP_FLAG=""
if [ "$QWEN35_FIX" = "1" ]; then MTP_FLAG="--no-mtp"; fi
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$GGUF_F16" --outtype f16 $MTP_FLAG \
  --model-name "$MODEL_NAME"

# Importance matrix: which weights matter most on real conversations, so 4-bit
# quantization spends its precision there. Calibrated on this model's own
# training conversations.
python scripts/make_calibration_text.py "$DATA_DIR/train.jsonl" "${GGUF_F16%.gguf}-calib.txt"
llama-imatrix -m "$GGUF_F16" -f "${GGUF_F16%.gguf}-calib.txt" -o "$IMATRIX" \
  -c 512 --chunks 150 -ngl "${IMATRIX_NGL:-99}"

# llama-quantize: brew install llama.cpp
# Q4_K_M by default (QUANT_TYPE overrides: IRx-mini uses Q8_0 -- at ~0.8B, 4-bit looped
# in multi-turn chats and Q8_0 passed). Q4_0 and IQ4_NL (faster prompt processing on Android's ARM CPUs)
# were tried for IRx-1 and failed scripts/check_gguf.py -- a 2B model looped in
# multi-turn chats at those formats; Q4_K_M passed.
llama-quantize --imatrix "$IMATRIX" "$GGUF_F16" "$GGUF_QUANT" "$QUANT_TYPE"

rm -f "$IMATRIX" "${GGUF_F16%.gguf}-calib.txt"
echo "GGUF ready: $GGUF_QUANT"
