---
license: apache-2.0
tags:
- gguf
- llama.cpp
- offline
- on-device
---

# IRx-1 (GGUF)

> Updated {{UPDATED}} · trained on news up to {{NEWS_DATE}} · rebuilt and re-checked every week

IRx-1 for phones and tablets — a small, fast member of the IRx family. Runs fully
offline in llama.cpp-based apps: PocketPal (iPhone, iPad, Android), LM Studio,
llama.rn / React Native apps and Ollama.

**Download:** `irx-1-Q4_K_M.gguf` (~1.2GB) — one file for every device. Runs on almost
any modern phone or tablet.

## The IRx family

All four run fully offline — nothing you type ever leaves your device.

| Model | Download | Best for | Phones & tablets (GGUF) | Mac (MLX) |
|---|---|---|---|---|
| **IRx-mini** | ~1.1GB | The smallest. Simple, quick answers on phones with just 4GB of RAM | [irx-mini-GGUF](https://huggingface.co/ikppramesh/irx-mini-GGUF) | [irx-mini](https://huggingface.co/ikppramesh/irx-mini) |
| **IRx-1** | ~1.2GB | Fast, quick everyday answers on almost any phone | [irx-1-GGUF](https://huggingface.co/ikppramesh/irx-1-GGUF) | [irx-1](https://huggingface.co/ikppramesh/irx-1) |
| **⭐ IRx-2 — Best Overall** | ~2.7GB | The best balance of answer quality and speed | [irx-2-GGUF](https://huggingface.co/ikppramesh/irx-2-GGUF) | [irx-2](https://huggingface.co/ikppramesh/irx-2) |
| **IRx-2 Pro** | ~5.3GB | The most detailed, long-form answers and long conversations | [irx-2-pro-GGUF](https://huggingface.co/ikppramesh/irx-2-pro-GGUF) | [irx-2-pro](https://huggingface.co/ikppramesh/irx-2-pro) |

**Not sure which to pick? Start with ⭐ IRx-2.** Choose IRx-mini for a 4GB-RAM phone,
IRx-1 for fast everyday answers on most phones, and IRx-2 Pro for the richest answers on
a device with 12GB+ RAM.

**See them side by side:** the [IRx board](https://claude.ai/artifact/1ha4zVccAwrKSm3mjKBLFU) compares all four models, shows
their measured speed live, and has an animated step-by-step setup guide for PocketPal
on iPhone and Android.

## What IRx-1 does best

IRx-1 is small and fast. It is built for quick,
everyday help: short answers, simple explanations, quick drafts and checklists,
instantly and offline — including places with no or poor internet.

| Who | Example things to ask |
|---|---|
| Software developers | "What does HTTP error 404 mean?" · "Write a Python one-liner to reverse a list." |
| Farmers | "Give me a simple way to track spraying dates for my crops." · "Draft a short message to my buyer about tomorrow's delivery." |
| Students | "Explain photosynthesis in two sentences." · "Give me 5 quiz questions on fractions." |
| Teachers | "Suggest 3 fun warm-up activities for a class of 10-year-olds." |
| Shop & small business owners | "Write a one-line WhatsApp message for my weekend sale." |
| Travellers | "Make a packing checklist for a 3-day trip." |
| Everyday life | "Suggest a quick breakfast." · "Convert 25°C to Fahrenheit." |

## Limitations

- **Facts can be wrong.** Like any small offline model it can state things confidently
  that aren't true — double-check anything important, such as prices, medicines and
  doses, laws, and local farming advice (seed varieties, chemical quantities, weather).
- **Not a replacement for professionals** — medical, legal and financial answers are
  general information only.
- **Don't enable native tool/function-calling in chat apps** — never trained; plain chat is reliable.

## Recommended settings (PocketPal and similar apps)

| Setting | Value |
|---|---|
| Context size | **8192** |
| Max new tokens | 1024 |
| Temperature | 0.7 |
| Top-P / Min-P | 0.95 / 0.05 |
| Repeat penalty | 1.1 |
| System prompt | optional — leave empty to use the built-in one |

A repeat penalty of 1.1 is also embedded in the file as the default; apps that let
you set your own sampling values (PocketPal does) need it set there too.

**Updating from an older download:** delete the old model in the app and download it
again — apps don't replace a downloaded file on their own. If you set a custom chat
template in the app, reset it so the one built into the file is used.

## Built-in chat template: thinking off, identity on

Apps built on llama.cpp often switch on a hidden "thinking" mode by default, which can
use up the whole conversation before any answer appears ("The conversation ran out of
room"). The template embedded in this file always keeps thinking off — the app's
reasoning toggle has no effect — and supplies the model's own system prompt when the
app sends none.

## Usage (command line)

```bash
llama-cli -m irx-1-Q4_K_M.gguf --jinja -c 8192 -p "How do I convert Celsius to Fahrenheit?"
```

No system prompt needed. Same capability/limitation notes as the
[main IRx-1 model card](https://huggingface.co/ikppramesh/irx-1) apply — small
model, not frontier-scale, don't expose tool/function-calling to it in host apps
that support that.

## Technical notes: a real conversion bug this build fixes

A stock `mlx_lm.convert` → `llama.cpp convert_hf_to_gguf.py` pipeline on this model
produces a GGUF that **loads without error but generates complete garbage** —
silently broken, not obviously broken. Found by directly diffing every tensor
between the raw HF checkpoint and MLX's converted output:

1. **`conv1d.weight` layout** — MLX stores it as (out, kernel, in); llama.cpp
   expects PyTorch's (out, in, kernel). Affects all 18 linear-attention layers'
   core recurrent-state computation.
2. **RMSNorm weight offset** — the raw checkpoint uses the
   zero-centered convention (multiplier = 1 + weight); MLX adds the 1.0 for its
   own kernel and that shifted value is what got exported. Affects 61 tensors
   (every `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and
   the final norm).

Both verified directly against the raw checkpoint (`mx.allclose` after undoing
each transform matches exactly) and fixed before conversion — see
`scripts/fix_gguf_mlx_conversion.py` in the
[main repo](https://github.com/ikppramesh/irx-1). Also needs `--no-mtp` at
convert time (this checkpoint doesn't carry an optional multi-token-prediction
head some conversion paths expect).

## Changelog

- **2026-11-04 — Reasoning, friendlier conversation, weekly updates**
  - **Step-by-step reasoning.** About 450 new worked maths and word-problem solutions,
    each checked against the correct answer before training (wrong ones discarded).
    Answers show numbered steps and end with a clear final answer, written in plain
    text rather than LaTeX so they display cleanly in any chat app.
  - **Warmer everyday conversation.** About 525 new examples of friendly, practical
    help: exams, interviews, family messages, budgeting, farming, health questions
    and more.
  - **News up to 4 November 2026.**
  - **Now rebuilt every week**, and each build must pass the behaviour check before
    it is published.

- **2026-10-03 — News refresh and quality update**
  - **Built-in news digest (up to 3 October 2026).** The model's built-in prompt now
    carries 15 of the day's top headlines across India, world, business, science and
    technology, AI and sports, so it can answer questions about them offline, in any
    app, with no internet. In testing, answers about digest headlines contained the
    real facts 47% of the time, against 0% for news learned through training alone.
  - **Knows its cutoff.** Asked how current it is, it says its news goes up to
    3 October 2026 and suggests checking a news source for anything later.
  - **News sources: 18.** Times of India, The Hindu (National, Business, Sport,
    Sci-Tech), Indian Express, NDTV, LiveMint, BBC (World, Business, Science,
    Technology), Al Jazeera, ESPNcricinfo, NASA, TechCrunch, The Verge and MIT
    Technology Review. Recent headlines were also added to training.
  - **Cleaner training data.** Removed answers that repeat themselves, refusals,
    answers copied from earlier in a chat and leftover chat-export text. Greetings no
    longer trigger a self-introduction, and identity answers never credit another AI
    company.
  - **Passed the pre-publish check:** repeated 8-turn conversations with the repeat
    penalty off and the app's thinking request on — no loops, no copied answers, no
    hidden reasoning, correct identity.
  - Keep in mind: news outside the digest is not reliably known, and small models can
    mix up names, numbers and dates. Double-check anything important. The digest adds
    a few seconds before the first reply of a new chat.

- **2026-09-30 (update)** — Fixed repeating/looping replies.
  - Symptom: in multi-turn chats the model repeated a word or sentence until the
    reply ran out ("pizza, pizza, pizza…"), or pasted an earlier answer again
    verbatim instead of answering the new question.
  - Cause: the GGUF was built by merging into the already-4-bit training base and
    quantizing to 4-bit a second time. The same fine-tune merged into the
    full-precision base and quantized once — with an importance matrix — shows
    neither problem.
  - Verified with an automated check (now run before every publish): 6 full
    8-turn conversations at repeat penalty 1.0 with the app's thinking request on
    — no loops, no copied answers, no hidden reasoning, correct identity.
  - Sampling defaults embedded in the file (temperature 0.7, top-p 0.95, min-p
    0.05, repeat penalty 1.1) for apps that use them.
  - Q4_0 withdrawn (looped in the same test); use Q4_K_M on every device.
- **2026-09-30** — Mobile fix release.
  - **Fixed empty replies / "The conversation ran out of room" in PocketPal** (and
    other llama.cpp apps): the embedded chat template now always turns thinking
    off. Verified with the app-default thinking request on: 0 reasoning tokens,
    answers start immediately (previously 1,000+ hidden reasoning tokens and no
    answer).
  - **Identity without a system prompt** — previously answered the base model's name
    when an app sent no system prompt; now answers as IRx-1 (new identity training
    examples + built-in default system prompt).
  - Retrained. Training examples are now fitted to the training window instead of
    being truncated mid-answer, and loss is computed on answers only. Scheduled
    retrains had been failing on a GPU out-of-memory error since 2026-09-11; fixed.

- **2026-09-08** — First working GGUF build. Two silent MLX→GGUF conversion bugs
  found and fixed (see above); verified generating correct, coherent output
  before publishing. Full model changelog (training rounds, RAG pipeline,
  etc.) on the [main model card](https://huggingface.co/ikppramesh/irx-1).

## License

Apache 2.0. This is a derivative fine-tuned model — full Apache 2.0 terms apply
as with any work under this license.
