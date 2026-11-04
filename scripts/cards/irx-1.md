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

![IRx-1](irx-1.png)

# IRx-1

> Updated {{UPDATED}} · trained on news up to {{NEWS_DATE}} · rebuilt and re-checked every week

A small, fast AI chat assistant that runs **entirely on your device** — no internet
connection, no account, and nothing you type ever leaves your phone, tablet or laptop.
IRx-1 is a light, fast member of the IRx family: quick everyday answers, even on
modest phones.

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

- **Not a frontier-scale model.** At ~2B parameters, it won't match large hosted models
  on hard multi-step reasoning, deep technical/coding problems, or breadth of world
  knowledge — that gap is a function of scale, not something fine-tuning erases
- **Not current-events aware on its own.** The model's weights are fixed as of
  training — by itself it has no access to recent news, prices, or events. Fine-tuning
  more often on news doesn't fix this reliably (news is dense with exactly the kind
  of fast-changing, precise facts fine-tuning is worst at — see the next point); the
  fix that's actually reliable is retrieval, not retraining, which the reference repo
  provides as an opt-in add-on (see Strengths) — the base model file here has no
  built-in awareness of anything past its training data.
- **Fine-tuning improves but doesn't guarantee facts.** Tested directly: adding
  accurate, spot-checked training examples measurably improved general-knowledge
  accuracy (most capitals, chemical symbols, and historical facts now answer
  correctly), but individual facts can still be wrong — especially where the base
  model already had a strong competing belief from pretraining (e.g. it still answers
  "Sahara" for the largest desert despite a correct "Antarctica" training example, and
  gives an imprecise population figure for India despite an accurate one in training).
  This is a real, partial improvement, not a fix — treat factual claims as
  worth double-checking, same as any small model.
- **Don't enable native tool/function-calling in chat apps.** The base architecture
  supports a native `<tool_call>` function-calling format, but IRx-1's fine-tuning
  never trained on or reinforced it. If a host app (LM Studio, Bionic, or similar)
  exposes tools/functions to the model, it can misfire the wrong tool for a plain
  question, or spiral into an unterminated loop hallucinating a fake multi-step
  agentic session — especially in apps with a large always-on tool/skill system
  prompt, which can overwhelm a model this size regardless of the question asked.
  Plain chat (no tools exposed) is reliable.
- **Occasional off answers at higher sampling temperatures** — like any small model,
  it can occasionally produce a repetitive or degenerate response; regenerating
  usually resolves it
- **Best suited to short-to-medium context** — very long, multi-document context
  windows aren't its strong suit
- **Facts can be wrong.** Like any small offline model it can state things confidently
  that aren't true — double-check anything important, such as prices, medicines and
  doses, laws, and local farming advice (seed varieties, chemical quantities, weather).
- **Not a replacement for professionals** — medical, legal and financial answers are
  general information only.

## How it was built

![Training pipeline](pipeline-diagram.png)

Full technical detail, including the exact scripts and the story behind each
data source, is in the [GitHub repo](https://github.com/ikppramesh/irx-1).

## Example queries

**"Who created you?"**
> I was created by Ramesh Inampudi from Hyderabad, India. He built me, IRx-1, at
> iramesh.com.

**"What questions should I ask before signing an apartment lease?"**
> 1. What is the total rent, including security deposit and utilities?
> 2. What are the rent payment terms, such as due dates and early termination fees?
> 3. What specific appliances and furniture are included in the lease?
> 4. Are there any pet fees or restrictions on pets?
> 5. What are the maintenance responsibilities for both the landlord and tenant?
> 6. What are the rules regarding subletting or assigning the lease to a new tenant?
> ...

**"What is the chemical symbol for gold?" / "Who painted the Mona Lisa?"**
> Au
>
> Leonardo da Vinci

**"Explain the difference between TCP and UDP in a couple of sentences."**
> TCP is a connection-oriented protocol that ensures reliable delivery by establishing a
> handshake, managing resources, and retransmitting lost packets, while UDP is a
> connectionless protocol that prioritizes speed over reliability, making it ideal for
> real-time applications like streaming and VoIP.

**"Write a short, encouraging note to leave for a roommate who is stressed about exams."**
> Hey, I know your brain is working overtime! Just remember to breathe, and we'll
> tackle these problems one at a time. You've got this, and I'm right here cheering
> you on.

(Real, unedited outputs. Small models vary run to run — regenerate if a particular
answer misses.)

## Tool-use example: xGAIR

[xGAIR](https://github.com/ikppramesh/XGAIR) is an MCP server that plugs AI coding
assistants into any GitHub repo. Its chat CLI matched only exact command syntax;
IRx-1 serves as an optional natural-language fallback, parsing free-form input into
the correct structured tool call. Real, verified outputs, repo names not seen
verbatim in training:

```
> hook up github.com/vercel/next.js
  → xgair_connect_repo { url: "github.com/vercel/next.js", repo: "next.js" }

> run discovery on stripe/stripe-node
  → xgair_discover_repo { repoId: "stripe/stripe-node" }

> check this snippet: DROP TABLE users;
  → xgair_validate { repoId: "", codeSnippet: "DROP TABLE users;" }
```

**Known limitation:** when a repo reference is embedded mid-sentence rather than at
the start of the message, the extracted `repoId` sometimes comes back empty even
though the tool choice itself is correct. Occasionally an off-topic message gets
mapped to a tool call instead of `{"tool": "unknown"}`. Neither is catastrophic by
design — the calling integration falls back to a current-repo context when `repoId`
is empty, and a wrongly-triggered call is a harmless read, not a destructive action.

## Current-events awareness: retrieval, not retraining

The weights above never change to add current-events knowledge — fine-tuning
doesn't reliably teach new facts (see Limitations), and news is the worst case
for that. The [GitHub repo](https://github.com/ikppramesh/irx-1) instead ships
an opt-in RAG pipeline:

![News RAG pipeline](rag-diagram.png)

Covers Indian news and, since AI-model releases go stale even faster than
general news, AI/tech news too — same reasoning, same mechanism. Two parallel
paths from the same source data: the left one (SQLite + FTS5) only works on
the Mac that built it. The right one is a free, static JSON snapshot
(GitHub Actions + Pages, no server) reachable from anywhere:

```
https://ikppramesh.github.io/irx-1/news.json
```

**This does not automatically make a mobile app "current."** The URL above
is a plain HTTP endpoint — something has to actually call it and search the
result. That's real client-side code, not a property of the model file. Full
working example (JS) in the
[GitHub repo](https://github.com/ikppramesh/irx-1#using-this-from-anywhere--mobile-or-anyone-elses-system).

**Honest result, not oversold:** narrow, specific questions ground correctly
("what did TechCrunch report about Meta's AI model?" answered accurately from
the actual retrieved article). Broad, open-ended questions don't — "what are
the latest AI models released?" still fell back to hallucinating stale,
made-up model names from frozen training memory, even with relevant articles
successfully retrieved and an explicit instruction to prefer them. A 2B model
juggling several simultaneous instructions (identity + grounding + retrieved
text) doesn't reliably prioritize all of them — the same pattern behind the
tool-calling/agentic-overwhelm limitation below. Ask specific questions.


## Usage (Mac, MLX)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("ikppramesh/irx-1")
messages = [{"role": "user", "content": "How do I convert Celsius to Fahrenheit?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024,
               sampler=make_sampler(temp=0.7, top_p=0.95)))
```

No system prompt is required: the built-in chat template supplies the IRx-1 one when none is given, and always keeps thinking mode off. Sample at a non-zero temperature
(e.g. `temp=0.7`). **On a phone or tablet**, use the
[GGUF build](https://huggingface.co/ikppramesh/irx-1-GGUF).

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

**2026-09-30**
- **Fixed empty replies / "The conversation ran out of room" on phones.** Apps built
  on llama.cpp (PocketPal and others) switch on the base architecture's thinking mode
  by default; the model then used its whole context on hidden reasoning before
  answering. The built-in chat template now always keeps thinking off (verified: 0
  reasoning tokens with the app default on).
- **Identity without a system prompt.** No training example had ever taught the name —
  it came only from a system prompt most apps never send, so the GGUF answered
  the base model's name. Added identity training examples and a built-in default system
  prompt; verified answering as IRx-1 with no system prompt, and with a generic one.
- Retrained. Examples are now fitted to the training window by dropping the oldest
  turns (previously truncated mid-answer, which teaches stopping early), and loss is
  computed on answers only.
- Fixed the scheduled retrain: every run since 2026-09-11 had crashed with a GPU
  out-of-memory error at the 1024-token training window. Now 768 (9.7GB peak).
- Fixed repeating/looping replies in the GGUF build (a word or sentence repeated
  until the reply ran out, or an earlier answer pasted again). Cause: the GGUF was
  merged into the already-4-bit training base and quantized a second time; it's
  now merged into the full-precision base and quantized once with an importance
  matrix. An automated multi-turn loop check now runs before every publish.

**2026-09-08**
- Automated recurring retrain + publish pipeline added: every 5 hours, after
  the news-fetch cycle, the model retrains on a small rolling set of
  news-derived examples and republishes both this repo and the GGUF repo.
  Tested end-to-end successfully. Documented limitation carried forward
  unchanged: this does not reliably teach the model new facts (see below) —
  the actual current-events answer path remains retrieval, not this.
- Made news retrieval portable beyond the build Mac — free GitHub Actions +
  Pages publish a JSON snapshot any client (mobile included) can fetch and
  search with no server; verified live and serving real data
- Wired news retrieval directly into the reference mobile app (RNFS-cached
  JSON, refreshed on launch and on demand — no model re-download required)
- Extended the news RAG with AI/tech feeds (TechCrunch AI, The Verge AI, MIT
  Technology Review), added Atom feed parsing, strengthened the grounding
  instruction — result honestly documented as mixed (see above), not oversold
- Rendered both architecture diagrams as actual images after finding Hugging
  Face's model card renderer doesn't support Mermaid (unlike GitHub)
- Model cards cleaned up: no base-model linkage, no license link naming
  anything specific — generic Apache 2.0 declaration only
- Added a creator-identity system prompt instruction, tested across phrasings
- Added the news RAG pipeline above — retrieval, not retraining; weights
  untouched
- Fixed GGUF export: two silent bugs in the MLX→GGUF conversion path (a
  conv1d weight axis-order mismatch, an RMSNorm weight offset convention
  mismatch) produced a file that loaded without error but generated complete
  garbage. Fixed and verified; working GGUF published to a separate repo.

**2026-09-07**
- Added a general-knowledge fine-tuning round (88 spot-checked
  geography/science/history examples, distilled from a larger local teacher)
  — verified genuine but partial accuracy improvement, documented honestly
  including facts that stayed wrong (see Limitations)
- Documented the tool/function-calling limitation
- Rebalanced the xGAIR intent-parsing training data (32 → 69 examples) after
  finding an earlier small "general knowledge about xGAIR" set was unreliable
  and caused hallucination; dropped it, kept the structured intent-parsing
  task that the data scale actually supports
- Initial release: personal-history + general-QA + distillation training
  pipeline, first published model card

## License

Apache 2.0. IRx-1 is a derivative fine-tuned model — full Apache 2.0 terms apply
as with any work under this license.
