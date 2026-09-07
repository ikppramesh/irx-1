#!/usr/bin/env bash
# Merge LoRA adapters into the base model, then quantize to GGUF for on-device use.
# Requires: source .venv/bin/activate, and llama.cpp cloned as a sibling dir for GGUF conversion.
set -euo pipefail

BASE_MODEL="mlx-community/Qwen3.5-2B-4bit"
ADAPTER_DIR="models/irx-1-adapters"
MERGED_DIR="models/irx-1-merged"
LLAMA_CPP_DIR="../llama.cpp"        # adjust or clone: git clone https://github.com/ggml-org/llama.cpp
GGUF_OUT="models/irx-1-qwen3.5-2b.gguf"
GGUF_QUANT_OUT="models/irx-1-qwen3.5-2b-Q4_K_M.gguf"

mlx_lm.fuse \
  --model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_DIR" \
  --save-path "$MERGED_DIR"

echo "Merged model written to $MERGED_DIR"
echo "Convert to GGUF with llama.cpp (not included here — clone separately):"
echo "  python $LLAMA_CPP_DIR/convert_hf_to_gguf.py $MERGED_DIR --outfile $GGUF_OUT"
echo "  $LLAMA_CPP_DIR/llama-quantize $GGUF_OUT $GGUF_QUANT_OUT Q4_K_M"
