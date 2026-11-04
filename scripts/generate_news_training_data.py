#!/usr/bin/env python3
"""
Turn recently fetched articles in data/news.db into a bounded set of training
examples (data/processed/distilled_news.jsonl).

Template-based, not teacher-generated: every answer is the fetched title/summary
itself, with its source and date, so nothing is paraphrased into new claims.

Three kinds of example:
  - "Tell me about: <headline>" -> summary, source and date
  - "What were the main <topic> headlines on <date>?" -> a short dated digest
  - "How up to date is your knowledge?" -> the news cutoff date, honestly stated

Each question has exactly one answer. The previous version asked "What's the
latest news from <source>?" with a different answer every time, which teaches
the model to answer that question by guessing.

Known limitation (documented in the model cards): fine-tuning does not reliably
teach specific, fast-changing facts. Retrieval at answer time
(scripts/news_context.py, docs/news.json) is the reliable current-events path.
This file is OVERWRITTEN on every run, so the mix reflects the latest articles.

Usage:
    python scripts/generate_news_training_data.py
"""

import argparse
import html
import json
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path("data/news.db")
OUT_PATH = Path("data/processed/distilled_news.jsonl")
IST = timezone(timedelta(hours=5, minutes=30))

TOPIC = {
    "Times of India": "India", "The Hindu": "India", "Indian Express": "India", "NDTV": "India",
    "LiveMint": "business", "The Hindu Business": "business", "BBC Business": "business",
    "BBC World": "world", "Al Jazeera": "world",
    "BBC Science": "science and technology", "BBC Technology": "science and technology",
    "The Hindu Sci-Tech": "science and technology", "NASA": "science and technology",
    "MIT Technology Review": "science and technology",
    "TechCrunch AI": "AI", "The Verge AI": "AI",
    "The Hindu Sport": "sports", "ESPNcricinfo": "sports",
}

CUTOFF_QUESTIONS = [
    "How up to date is your knowledge?",
    "What is the latest news you know about?",
    "Do you know about recent events?",
    "Until when is your information current?",
    "Are you connected to the internet?",
    "Can you tell me today's news?",
]


def clean(text):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def day(ts):
    d = datetime.fromtimestamp(ts, IST)
    return f"{d.day} {d.strftime('%B %Y')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--max-articles", type=int, default=150,
                        help="headline examples, spread evenly across sources")
    parser.add_argument("--digest-size", type=int, default=5)
    # 3 variants with a stronger fine-tune raised recall only 0% -> 12% on IRx-mini and made
    # it loop (failed scripts/check_gguf.py); 1 keeps training gentle. Current news reaches
    # the models through the digest in their chat template instead (scripts/make_news_digest.py).
    parser.add_argument("--variants", type=int, default=1, help="questions per headline (1-3)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.db.exists():
        print(f"No news index at {args.db} -- run scripts/fetch_news.py first. Skipping.")
        args.out.write_text("")
        return

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT source, title, summary, fetched_at FROM articles ORDER BY fetched_at DESC"
    ).fetchall()
    conn.close()
    if not rows:
        args.out.write_text("")
        return

    seen, by_source = set(), defaultdict(list)
    for source, title, summary, ts in rows:
        title, summary = clean(title), clean(summary)[:400]
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        by_source[source].append((title, summary, ts))

    examples = []

    # Headlines: round-robin across sources, newest first, so no single feed dominates.
    picked, queues = [], {s: list(v) for s, v in by_source.items()}
    while len(picked) < args.max_articles and any(queues.values()):
        for s in sorted(queues):
            if queues[s] and len(picked) < args.max_articles:
                picked.append((s, *queues[s].pop(0)))
    for source, title, summary, ts in picked:
        if not summary or summary.lower() == title.lower():
            continue
        answer = f"{summary} (Reported by {source} on {day(ts)}.)"
        examples.append({"messages": [
            {"role": "user", "content": f"Tell me about this news: {title}"},
            {"role": "assistant", "content": answer},
        ]})
        # Same fact behind different cues: asked once, a headline is not recalled at all
        # (measured 0% with scripts/check_news_recall.py).
        cue = " ".join(title.split()[:5])
        variants = [
            (f"What happened with {cue}?", f"{title}. {answer}"),
            (f"Any recent news on {cue}?", f"Yes — {title}. {answer}"),
        ]
        for q, a in variants[: max(0, args.variants - 1)]:
            examples.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    # Daily digests per topic.
    digest = defaultdict(list)
    for source, items in by_source.items():
        for title, _, ts in items:
            digest[(TOPIC.get(source, "general"), day(ts))].append((title, source))
    rng = random.Random(args.seed)
    for (topic, date), items in sorted(digest.items()):
        if len(items) < 3:
            continue
        chosen = rng.sample(items, min(args.digest_size, len(items)))
        lines = "\n".join(f"- {t} ({s})" for t, s in chosen)
        examples.append({"messages": [
            {"role": "user", "content": f"What were the main {topic} headlines on {date}?"},
            {"role": "assistant", "content": f"Some of the {topic} headlines from {date}:\n{lines}"},
        ]})

    # Honest knowledge cutoff.
    cutoff = day(max(ts for _, _, _, ts in rows))
    # Repeated with three phrasings: asked once each, the smallest model didn't learn it.
    cutoff_answers = [
        f"I run fully offline, so I can't look anything up. My training includes news up to {cutoff}. For anything after that, or to double-check details, please check a news source.",
        f"My news knowledge goes up to {cutoff}. I can't browse the internet, so for anything more recent please check a news source.",
        f"I was trained with news up to {cutoff}. I work offline, so I don't know about events after that date.",
    ]
    for q in CUTOFF_QUESTIONS:
        for a in cutoff_answers:
            examples.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples ({len(picked)} headlines considered, "
          f"{sum(1 for k, v in digest.items() if len(v) >= 3)} daily digests, news up to {cutoff}) -> {args.out}")


if __name__ == "__main__":
    main()
