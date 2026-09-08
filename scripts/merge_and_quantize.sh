#!/usr/bin/env bash
# Merge LoRA adapters into the base model, then quantize to GGUF for on-device use
# (llama.cpp-based runtimes: LM Studio, llama.rn mobile apps, Ollama import, etc).
# Requires: source .venv/bin/activate, and llama.cpp cloned as a sibling dir:
#   git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
#   pip install -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
#     (in a SEPARATE venv, or expect it to downgrade transformers/numpy/tokenizers
#      here — it did, and broke mlx_lm's tokenizer loading, the first time this
#      was tried; restore this project's requirements.txt afterward if so)
set -euo pipefail

BASE_MODEL="mlx-community/Qwen3.5-2B-4bit"
ADAPTER_DIR="models/irx-1-adapters"
MERGED_DIR="models/irx-1-merged-fp16"
LLAMA_CPP_DIR="../llama.cpp"
GGUF_F16="models/irx-1-qwen3.5-2b-f16.gguf"
GGUF_QUANT="models/irx-1-qwen3.5-2b-Q4_K_M.gguf"

mlx_lm.fuse \
  --model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_DIR" \
  --save-path "$MERGED_DIR" \
  --dequantize

# Two real bugs in mlx_lm.convert's Qwen3.5 export silently corrupt GGUF
# conversion — the file converts and loads without any error, but generates
# complete garbage. Found by diffing every tensor against the raw HF
# checkpoint; see the docstring in fix_gguf_mlx_conversion.py for the full
# story (conv1d.weight layout + a Gemma-style RMSNorm weight offset).
python scripts/fix_gguf_mlx_conversion.py "$MERGED_DIR"

# mlx_lm.fuse's --dequantize also writes an auto-generated README.md into
# $MERGED_DIR with `base_model: mlx-community/Qwen3.5-2B-4bit` and a Qwen
# license link in its frontmatter. convert_hf_to_gguf.py reads exactly that
# file as this model's "model card" and copies those fields straight into
# the GGUF's general.base_model.* / general.license.link metadata -- which
# is what Hugging Face and GGUF-reading apps (LM Studio, Ollama, etc.) then
# surface as "fine-tuned from Qwen3.5-2B". Deleting it here (it's a
# throwaway intermediate anyway, not the published README) stops that
# heuristic from firing; --model-name and --metadata below set the fields
# we actually want instead.
rm -f "$MERGED_DIR/README.md"

# --no-mtp: this checkpoint doesn't carry Qwen3.5's optional multi-token-
# prediction head (mlx_lm.convert drops it; it's not needed for normal,
# non-speculative-decoding inference).
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$GGUF_F16" --outtype f16 --no-mtp \
  --model-name "IRx-1"

# llama-quantize: brew install llama.cpp
llama-quantize "$GGUF_F16" "$GGUF_QUANT" Q4_K_M

echo "GGUF ready: $GGUF_QUANT"
