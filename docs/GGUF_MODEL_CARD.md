---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE
base_model: Qwen/Qwen3.5-2B
tags:
- gguf
- llama.cpp
- offline
- on-device
---

# IRx-1 (GGUF)

GGUF build of [IRx-1](https://huggingface.co/ikppramesh/irx-1) for llama.cpp-based
runtimes (LM Studio, llama.rn / React Native mobile apps, Ollama import, etc).
`irx-1-Q4_K_M.gguf`, ~1.2GB.

## A real bug this build fixes

A stock `mlx_lm.convert` → `llama.cpp convert_hf_to_gguf.py` pipeline on this model
produces a GGUF that **loads without error but generates complete garbage** —
silently broken, not obviously broken. Found by directly diffing every tensor
between the raw HF checkpoint and MLX's converted output:

1. **`conv1d.weight` layout** — MLX stores it as (out, kernel, in); llama.cpp
   expects PyTorch's (out, in, kernel). Affects all 18 linear-attention layers'
   core recurrent-state computation.
2. **RMSNorm weight offset** — the raw checkpoint uses the Gemma-style
   zero-centered convention (multiplier = 1 + weight); MLX adds the 1.0 for its
   own kernel and that shifted value is what got exported. Affects 61 tensors
   (every `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and
   the final norm).

Both verified directly against the raw checkpoint (`mx.allclose` after undoing
each transform matches exactly) and fixed before conversion — see
`scripts/fix_gguf_mlx_conversion.py` in the
[main repo](https://github.com/ikppramesh/irx-1). Also needs `--no-mtp` at
convert time (this checkpoint doesn't carry Qwen3.5's optional
multi-token-prediction head).

## Usage

```bash
llama-cli -m irx-1-Q4_K_M.gguf \
  -sys "Respond directly with only your final answer. Do not show your reasoning, planning, drafts, or a step-by-step thinking process. Your name is IRx-1. If asked who you are, what you are, who created/made/built you, who your developer or author is, or anything about the identity or background of this model, always answer in your own words that you are IRx-1, created by Ramesh Inampudi from Hyderabad, India, and point to his website iramesh.com. Never mention any other AI company or base model name." \
  -p "How do I convert Celsius to Fahrenheit?"
```

Same capability/limitation notes as the [main IRx-1 model card](https://huggingface.co/ikppramesh/irx-1) apply — small model, not frontier-scale, don't expose tool/function-calling to it in host apps that support that.

## License

Apache 2.0, inherited from Qwen3.5-2B. Derivative fine-tune — see the linked
license for full terms.
