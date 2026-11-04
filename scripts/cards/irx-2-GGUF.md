---
license: apache-2.0
tags:
- gguf
- llama.cpp
- offline
- on-device
---

# IRx-2 (GGUF) — ⭐ Best Overall

> Updated {{UPDATED}} · trained on news up to {{NEWS_DATE}} · rebuilt and re-checked every week

The recommended model of the IRx family: a private, offline AI chat assistant with the
best balance of answer quality and speed. Everything runs on your device — no
internet connection, no account, and nothing you type ever leaves your phone, tablet
or laptop. This is the build for phones, tablets and llama.cpp-based apps (PocketPal, LM Studio, llama.rn, Ollama).

**Download:** `irx-2-Q4_K_M.gguf` (~2.7GB) — one file for iPhone, iPad, Android and Mac. Uses about 3.5GB of RAM while running, so it fits phones and tablets with 8GB+ RAM. On phones with 6GB or less, use [IRx-1](https://huggingface.co/ikppramesh/irx-1-GGUF).

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

## What IRx-2 does best — ⭐ Best Overall

IRx-2 is the recommended model for most people. It is about twice the size of IRx-1,
which shows in clearer reasoning, better-structured answers, more reliable writing and
stronger coding help — while still running comfortably on phones and tablets with
8GB+ RAM.

| Who | Example things to ask |
|---|---|
| Software developers | "Explain this error and how to fix it: …" · "Write a function that validates an email address, with tests." · "Why is this SQL query slow?" |
| Farmers | "Plan a monthly budget for a 2-acre vegetable farm." · "Compare drip and flood irrigation: pros and cons." · "Write a loan application letter to my bank." |
| Students | "Make a one-week study plan for my exams." · "Explain Newton's laws with everyday examples." |
| Teachers | "Create a 40-minute lesson plan on the water cycle, with a short quiz." |
| Shop & small business owners | "Outline a simple business plan for a tea stall." · "Write a polite reply to a customer complaint." |
| Writers & creators | "Outline a 5-minute YouTube script about saving money." · "Make this paragraph sound more professional." |
| Job seekers | "Improve these resume bullet points." · "Give me 10 likely interview questions for a sales role." |
| Everyday life | "Plan a family weekend on a budget." · "Help me write a birthday message for my father." |

Measured on the same Mac CPU: IRx-2 generates about 26 tokens/s vs IRx-1's 54 — roughly half the speed, for noticeably better answers.

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
llama-cli -m irx-2-Q4_K_M.gguf --jinja -c 8192 -p "How do I convert Celsius to Fahrenheit?"
```

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

- **2026-09-30** — First release. Built with the IRx pipeline's fixes: fine-tune merged
  into the full-precision base and quantized once with an importance matrix (no
  repeating/looping replies), thinking mode always off, identity built in. Passed the
  automated pre-publish check: repeated 8-turn conversations with the repeat penalty
  off and the app's thinking request on — no loops, no copied answers, no hidden
  reasoning, correct identity.

## License

Apache 2.0. IRx-2 is a derivative fine-tuned model — full Apache 2.0 terms apply as with
any work under this license.
