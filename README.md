<div align="center">

<img src="irx-1.png" alt="IRx-1" width="480" />

**A ~2B parameter chat model, fine-tuned for fast, private, fully offline use on personal devices.**

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ikppramesh%2Firx--1-yellow)](https://huggingface.co/ikppramesh/irx-1)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](docs/LICENSE_NOTE.md)
[![Built with MLX](https://img.shields.io/badge/built%20with-MLX-black)](https://github.com/ml-explore/mlx-lm)

</div>

---

## What is IRx-1?

IRx-1 is a small, self-contained language model built end-to-end on a single Apple
Silicon Mac — no cloud GPUs, no massive pretraining run. It's fine-tuned (LoRA/QLoRA)
from an open-weight ~2B base model on a deliberately mixed dataset: personal
conversation history for style, a public instruction dataset for general-purpose
Q&A breadth, and hand-authored data teaching it to serve as the natural-language
front-end for a real developer tool ([xGAIR](https://github.com/ikppramesh/XGAIR)).

The result runs entirely on-device — quantized to 4-bit, no internet connection
required at inference time — while still being genuinely useful for everyday tasks.

**[→ Try it on Hugging Face](https://huggingface.co/ikppramesh/irx-1)**

## Strengths

- **Fast and lightweight** — small enough to run offline on a phone or laptop
- **Private by design** — nothing you type ever leaves the device
- **Personalized style** — fine-tuned on real conversational data, not generic
  boilerplate
- **Broad everyday usefulness** — Q&A, writing help, planning, explanations,
  brainstorming, classification, summarization
- **Tool-aware** — can parse free-form natural language into structured commands for
  a real integrated developer tool (see [xGAIR integration](#real-world-integration-xgair))

## Limitations

- Not a frontier-scale model — won't match large hosted models on hard multi-step
  reasoning, deep technical/coding problems, or breadth of world knowledge; that gap
  is a function of parameter count and pretraining scale, not something fine-tuning
  erases
- Not current-events aware — its knowledge is fixed as of training, with no live or
  retrieval-based access to recent news, prices, or events
- Like any small model, occasional repetitive or off answers at higher sampling
  temperatures; regenerating usually resolves it

## Quick start

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("ikppramesh/irx-1")

messages = [
    {"role": "system", "content": "Respond directly with only your final answer. "
     "Do not show your reasoning, planning, drafts, or a step-by-step thinking process."},
    {"role": "user", "content": "How do I convert Celsius to Fahrenheit?"},
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

The system prompt matters — without it the model can surface visible "thinking"
narration instead of a direct answer. Sample at a non-zero temperature (e.g.
`temp=0.7`); greedy decoding (`temp=0`) is prone to repetition loops on a model
this size.

## How it was built

```
data/raw/  ──────────────►  scripts/prepare_data.py  ─────┐
(personal history export,       (redact + exclude PII)     │
 gitignored, never committed)                               │
                                                              ▼
data/seed_prompts.txt  ─►  scripts/generate_distillation_data.py ─┐
                              (teacher model, local)                │
databricks-dolly-15k  ───►  scripts/prepare_general_data.py ───────┤──►  scripts/combine_data.py
                              (balanced general Q&A sample)         │        (train/valid split)
xGAIR docs/source  ──────►  scripts/prepare_xgair_data.py ─────────┘              │
  (hand-authored, grounded                                                        ▼
   in verified source)                                                  scripts/finetune.sh
                                                                          (QLoRA, MLX, on-device)
                                                                                    │
                                                                                    ▼
                                                                      scripts/merge_and_quantize.sh
                                                                        (fuse adapters → GGUF)
                                                                                    │
                                                                                    ▼
                                                                          Hugging Face + GitHub
```

1. **Personal data** (`scripts/prepare_data.py`) — a claude.ai conversation export is
   parsed into instruction/response pairs, each capped to a 6-message context window.
   Before training, every message is scrubbed for credentials, API keys, tokens,
   connection strings, and PII via regex — and, since regex can't safely catch
   everything (full names in non-Latin scripts, birth/health/identity details), entire
   conversation topics are excluded rather than partially redacted, verified with a
   full-text scan across the whole export, not just conversation titles.
2. **Style distillation** (`scripts/generate_distillation_data.py`) — a separate local
   model is prompted over ~50 diverse everyday prompts with reasoning-narration
   suppressed, producing a clean formatting/quality baseline.
3. **General knowledge breadth** (`scripts/prepare_general_data.py`) — a balanced
   400-example sample across 8 task categories from
   [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
   (CC-BY-SA-3.0, human-written, not AI-generated).
4. **Tool-use skill** (`scripts/prepare_xgair_data.py`) — hand-authored examples
   teaching IRx-1 to parse free-form language into structured tool calls for xGAIR's
   chat router, plus general knowledge about what xGAIR does — grounded directly in
   verified source, not freely generated, to avoid hallucinated tool schemas.
5. **Fine-tuning** (`scripts/finetune.sh`) — QLoRA (4-bit base + LoRA adapters on 4
   layers), via [MLX](https://github.com/ml-explore/mlx-lm), entirely on a single
   Apple Silicon machine.
6. **Merge & quantize** (`scripts/merge_and_quantize.sh`) — LoRA adapters fused back
   into the base weights for a single self-contained checkpoint.
7. **Serve** (`scripts/serve.sh`, `scripts/chat.py`) — an OpenAI-compatible local HTTP
   API or an interactive terminal chat, both fully offline.

## Real-world integration: xGAIR

[xGAIR](https://github.com/ikppramesh/XGAIR) is an MCP (Model Context Protocol) server
that plugs AI coding assistants into any GitHub repo with structured, context-aware
development intelligence. Its chat CLI matched only exact command syntax (regex).
IRx-1 now serves as an optional natural-language fallback: free-form input that
doesn't match an exact pattern is parsed by IRx-1 (running locally, fully offline)
into the correct structured tool call.

```
> can you fix the login bug in acme/widgets
  → xgair_start_task { repoId: "acme/widgets", taskDescription: "fix the login bug", taskType: "fix" }
```

See xGAIR's README, "Stage 2b — Natural-Language Fallback via IRx-1", for the full
integration.

## Project structure

```
IRx-1/
├── data/
│   ├── raw/              # personal export — gitignored
│   ├── processed/        # generated training JSONL — gitignored
│   └── seed_prompts.txt  # style-distillation seed prompts
├── docs/
│   └── LICENSE_NOTE.md   # license/attribution details for all training data sources
├── models/                # trained weights — gitignored (published to Hugging Face instead)
├── scripts/
│   ├── prepare_data.py            # personal history → redacted training data
│   ├── generate_distillation_data.py
│   ├── prepare_general_data.py    # databricks-dolly-15k sample
│   ├── prepare_xgair_data.py      # xGAIR tool-use + knowledge data
│   ├── combine_data.py            # merge all sources → train/valid split
│   ├── finetune.sh                # QLoRA fine-tune
│   ├── merge_and_quantize.sh      # fuse adapters, quantize
│   ├── serve.sh                   # local OpenAI-compatible API
│   └── chat.py                    # interactive terminal chat
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ikppramesh/irx-1.git
cd irx-1
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Model weights aren't in this repo (too large for git, and Hugging Face is the right
home for them) — pull the published model directly:

```bash
python3 scripts/chat.py --model ikppramesh/irx-1
```

To reproduce training, you'll need your own `data/raw/conversations.json`
(claude.ai → Settings → Account → Export data) — everything downstream is scripted.

## License

Apache 2.0. IRx-1 is a derivative fine-tune of an open-weight base model, plus data
from databricks-dolly-15k (CC-BY-SA-3.0). Full attribution and lineage details in
[`docs/LICENSE_NOTE.md`](docs/LICENSE_NOTE.md).
