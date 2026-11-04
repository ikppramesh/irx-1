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

# IRx-mini

> Updated {{UPDATED}} · trained on news up to {{NEWS_DATE}} · rebuilt and re-checked every week

The smallest member of the IRx family: a private, offline AI chat assistant made to run
on phones with just 4GB of RAM. Everything runs on your device — no internet
connection, no account, and nothing you type ever leaves your phone.

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

## What IRx-mini does best

IRx-mini is the smallest member of the family, made for phones with only **4GB of
RAM**. It gives short, simple answers, quick explanations and everyday help, fully
offline, on phones that can't fit the larger models.

| Who | Example things to ask |
|---|---|
| Farmers | "When should I water tomato plants?" · "Write a short message to my buyer." |
| Students | "What is photosynthesis? Explain simply." · "5 quick quiz questions on fractions." |
| Shop & small business owners | "Write a one-line offer for my shop." · "How do I calculate 15% profit?" |
| Software developers | "What does HTTP error 404 mean?" |
| Travellers | "Make a short packing list for 2 days." |
| Everyday life | "Convert 25°C to Fahrenheit." · "Suggest a quick breakfast." |

## Why a model for 4GB phones?

Many phones in use today, especially budget Android phones and older models, have
only 4GB of RAM. The phone's own system and background apps already use a large part
of that, so an AI model has to be small to load and run reliably — the larger IRx
models are simply too big for these phones.

IRx-mini is built for exactly that:

- **Small enough to fit.** About 1.1GB to download and roughly 1.3GB of memory
  while running, leaving room for the phone's system and other apps.
- **Works with no internet.** Useful where connectivity is weak, slow or expensive —
  on farms, in villages, while travelling, or on a limited data plan. Nothing is
  sent to the cloud, so there are no data costs and no waiting on a network.
- **Private.** Everything stays on the phone.
- **Light on storage and battery.** A small model loads quickly and does less work
  per answer.

The trade-off is depth: IRx-mini gives shorter, simpler answers and makes more
mistakes than the larger models. If your phone has 8GB+ RAM, ⭐ IRx-2 will give much
better answers.

Measured on the same Mac CPU: IRx-mini generates about 89 tokens/s — the fastest of the family (IRx-1: about 51 in the same test).

## Usage (Mac, MLX)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("ikppramesh/irx-mini")
messages = [{"role": "user", "content": "How do I convert Celsius to Fahrenheit?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024,
               sampler=make_sampler(temp=0.7, top_p=0.95)))
```

No system prompt is required: IRx-mini's identity is trained into the model, and the built-in chat template always keeps thinking mode off. Sample at a non-zero temperature
(e.g. `temp=0.7`). **On a phone or tablet**, use the
[GGUF build](https://huggingface.co/ikppramesh/irx-mini-GGUF).

## Limitations

- **Not a frontier-scale model** — it won't match large hosted AI services on very hard
  multi-step reasoning, deep coding problems or breadth of world knowledge.
- **Facts can be wrong.** Like any small offline model it can state things confidently
  that aren't true — double-check anything important, such as prices, medicines and
  doses, laws, and local farming advice (seed varieties, chemical quantities, weather).
- **Not a replacement for professionals** — medical, legal and financial answers are
  general information only.
- **Limited news knowledge.** IRx-mini was trained on headlines up to {{NEWS_DATE}},
  but at this size it recalls news poorly and may even give the wrong cutoff date. For
  recent news, use ⭐ IRx-2 or check a news source.
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

- **2026-10-03 — Rebuilt: no more looping on long answers.** The first release could
  repeat a phrase endlessly on long, structured answers such as day-by-day trip
  itineraries in PocketPal, whose default repeat penalty is off. IRx-mini is now built on
  a stronger 1B base:
  - **Long answers:** tested with trip itineraries (Lakshadweep, Kerala, Rajasthan) with the
    repeat penalty off and on, plus repeated 8-turn chats: no loops.
  - **Size and speed:** the download is about 1.1GB (was 0.8GB), still a fit for 4GB-RAM
    phones, and it is faster: about 89 tokens/s on the Mac CPU test (was 60–74).
  - **Answer quality:** answers are noticeably more complete and better structured,
    including code.
  - **Identity:** IRx-mini's name and creator are now trained into the model, with no
    system prompt added to your messages.
  - **News:** the built-in news digest was removed from IRx-mini, because at this size
    the long prompt pulled answers off-topic. News recall is limited (see Limitations).
  - **Updating:** delete the old IRx-mini in your app and download it again.
- **2026-10-03** — First release, trained on news up to 3 October 2026 from 18
  sources (see the news update notes on the other IRx cards). Built with the IRx pipeline's fixes: fine-tune merged
  into the full-precision base and quantized once with an importance matrix, thinking
  mode always off, identity built in. Passed the automated pre-publish check:
  repeated 8-turn conversations with the repeat penalty off and the app's thinking
  request on — no loops, no copied answers, no hidden reasoning, correct identity.

## License

Apache 2.0. IRx-mini is a derivative fine-tuned model — full Apache 2.0 terms apply as with
any work under this license.
