---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
thumbnail: https://huggingface.co/ikppramesh/irx-1/resolve/main/irx-1.png
tags:
- mlx
- lora
- offline
- on-device
---

# IRx-2 Pro

> Updated {{UPDATED}} · trained on news up to {{NEWS_DATE}} · rebuilt and re-checked every week

The largest edition of the IRx family: a private, offline AI chat assistant for long,
detailed answers and extended conversations. Everything runs on your device — no
internet connection, no account, and nothing you type ever leaves it.

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

## What IRx-2 Pro does best

IRx-2 Pro is the largest edition. It was trained with the longest context window of
the family (so it learned from nearly all of the long conversations in the training
data), which makes it the best choice for long, detailed answers and for working
through a topic over many messages. It runs at about the same speed as IRx-2, but
needs more memory: use it on devices with 12GB+ RAM, or on a Mac.

| Who | Example things to ask |
|---|---|
| Software developers | "Help me design a small inventory app step by step — data model, screens and API." · "Walk me through refactoring this code over a few messages." |
| Farmers | "Help me write a full-season plan for my farm — sowing, watering, spraying and harvest — then let's refine it." · "Write a detailed proposal to sell my produce to a local supermarket." |
| Students & researchers | "Turn these notes into a structured report with headings and a summary." · "Explain this topic in depth, then quiz me on it." |
| Teachers | "Create a full week's unit on fractions: daily lessons, activities and a final test." |
| Business owners | "Draft a detailed marketing plan for my new bakery." · "Write a complete employee handbook outline." |
| Writers & creators | "Write a 1,500-word blog post on healthy habits." · "Help me develop a short story over several messages." |
| Everyday life | "Plan a 7-day family trip with a day-by-day schedule and budget." |

Measured on the same Mac CPU: IRx-2 Pro generates about 24 tokens/s, close to IRx-2's 26 — most of its extra size is lookup tables that cost little compute.

## Usage (Mac, MLX)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("ikppramesh/irx-2-pro")
messages = [{"role": "user", "content": "How do I convert Celsius to Fahrenheit?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024,
               sampler=make_sampler(temp=0.7, top_p=0.95)))
```

No system prompt is required: the built-in chat template supplies the IRx-2 Pro one when none is given, and always keeps thinking mode off. Sample at a non-zero temperature
(e.g. `temp=0.7`). **On a phone or tablet**, use the
[GGUF build](https://huggingface.co/ikppramesh/irx-2-pro-GGUF).

## Limitations

- **Not a frontier-scale model** — it won't match large hosted AI services on very hard
  multi-step reasoning, deep coding problems or breadth of world knowledge.
- **Facts can be wrong.** Like any small offline model it can state things confidently
  that aren't true — double-check anything important, such as prices, medicines and
  doses, laws, and local farming advice (seed varieties, chemical quantities, weather).
- **Not a replacement for professionals** — medical, legal and financial answers are
  general information only.
- **News up to {{NEWS_DATE}} only** — it can't look anything up, and news details
  can be imprecise.
- **Don't enable native tool/function-calling in chat apps** — never trained; plain
  chat is reliable.

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

- **2026-09-30** — First release (published earlier the same day under a different
  repository name; old links redirect here). Trained with a 2048-token window, merged
  into the full-precision base and quantized once with an importance matrix. Passed the
  automated pre-publish check: repeated 8-turn conversations with the repeat penalty off
  and the app's thinking request on — no loops, no copied answers, no hidden reasoning,
  correct identity.

## License

Apache 2.0. IRx-2 Pro is a derivative fine-tuned model — full Apache 2.0 terms apply as with
any work under this license.
