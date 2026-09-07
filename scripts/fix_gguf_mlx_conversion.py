#!/usr/bin/env python3
"""
Fix two real bugs that silently corrupt GGUF conversion of any Qwen3.5 MLX
checkpoint. Both were found by directly comparing every tensor in the raw
Qwen/Qwen3.5-2B checkpoint against mlx_lm.convert's output: conversion
"succeeds" and the resulting GGUF loads without error, but generation is
complete garbage, because these two silent transforms MLX applies for its
own kernels never get undone on export.

1. conv1d.weight layout: MLX's convention is (out_channels, kernel_size,
   in_channels); llama.cpp's converter expects PyTorch's (out_channels,
   in_channels, kernel_size). Affects `linear_attn.conv1d.weight` in all
   18 linear-attention layers. Verified: transposing axes (0, 2, 1) back
   reproduces the raw checkpoint's values exactly (mx.allclose == True).

2. RMSNorm weight offset: the raw checkpoint stores the Gemma-style
   zero-centered offset (actual multiplier = 1 + weight); MLX's qwen3_5
   implementation adds the 1.0 internally for its own forward pass and
   that shifted value is what gets saved on export. Affects
   input_layernorm, post_attention_layernorm, self_attn.q_norm,
   self_attn.k_norm, and the final model norm -- 61 tensors, every one
   differing from the raw checkpoint by ~1.0. Verified: mlx_value - 1.0
   reproduces the raw checkpoint's values (mx.allclose == True).

Usage:
    python scripts/fix_gguf_mlx_conversion.py <merged_model_dir>
"""

import os
import re
import sys
from pathlib import Path

import mlx.core as mx

NORM_PATTERN = re.compile(
    r"(input_layernorm|post_attention_layernorm|self_attn\.(q|k)_norm|(?<!\.)norm)\.weight$"
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/fix_gguf_mlx_conversion.py <merged_model_dir>")
        sys.exit(1)

    merged_dir = Path(sys.argv[1])
    weights_path = merged_dir / "model.safetensors"

    tensors = mx.load(str(weights_path))

    conv_fixed = [k for k in tensors if k.endswith("linear_attn.conv1d.weight")]
    for key in conv_fixed:
        tensors[key] = mx.transpose(tensors[key], (0, 2, 1))

    norm_fixed = [
        k for k in tensors
        if k.endswith("input_layernorm.weight")
        or k.endswith("post_attention_layernorm.weight")
        or k.endswith("self_attn.q_norm.weight")
        or k.endswith("self_attn.k_norm.weight")
        or k.endswith(".model.norm.weight")
        or k == "language_model.model.norm.weight"
    ]
    for key in norm_fixed:
        tensors[key] = tensors[key].astype(mx.float32) - 1.0
        tensors[key] = tensors[key].astype(mx.bfloat16)

    tmp_path = weights_path.with_name("tmp_gguf_fix.safetensors")
    mx.save_safetensors(str(tmp_path), tensors)
    os.replace(tmp_path, weights_path)

    print(f"Fixed conv1d layout on {len(conv_fixed)} tensors")
    print(f"Fixed norm-weight offset on {len(norm_fixed)} tensors")
    print(f"-> {weights_path}")


if __name__ == "__main__":
    main()
