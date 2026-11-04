#!/usr/bin/env python3
"""
Build the compact news digest that goes into each model's built-in system prompt.

Fine-tuning does not make a small model recall news (measured: 0% recall after
training, 12% with an aggressive fine-tune that also made it loop). Headlines
placed in the prompt are answered from context instead, offline, in any app.

Picks the newest headlines per topic (India, world, business, science and
technology, AI, sports), a few each, and writes {"cutoff", "text"} as JSON for
scripts/make_chat_template.py --digest-file. Kept on one line and free of quote,
brace and percent characters so it embeds safely in a Jinja string literal.

Usage:
    python scripts/make_news_digest.py --out data/news_digest.json
"""

import argparse
import json
import sqlite3
from pathlib import Path

from generate_news_training_data import TOPIC, clean, day

ORDER = ["India", "world", "business", "science and technology", "AI", "sports"]
# Recurring filler rather than news.
SKIP_PREFIXES = ("gold rate", "apod", "watch:", "tech now", "live:", "in pics", "photos:")


def safe(text):
    text = text.replace('"', "'").replace("“", "'").replace("”", "'")
    for ch in "{}%\\\n\r":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/news.db")
    parser.add_argument("--out", default="data/news_digest.json")
    parser.add_argument("--per-topic", type=int, default=3)
    args = parser.parse_args()

    rows = sqlite3.connect(args.db).execute(
        "SELECT source, title, fetched_at FROM articles ORDER BY fetched_at DESC").fetchall()
    cutoff = day(rows[0][2])
    picked, seen = {t: [] for t in ORDER}, set()
    for source, title, ts in rows:
        topic, title = TOPIC.get(source), safe(clean(title))
        if (topic not in picked or not title or title.lower() in seen or len(title.split()) < 5
                or title.lower().startswith(SKIP_PREFIXES) or " live" in title.lower()[:40]):
            continue
        if len(picked[topic]) < args.per_topic and source not in {s for _, s, _ in picked[topic]}:
            picked[topic].append((title, source, day(ts)))
            seen.add(title.lower())
        if all(len(v) >= args.per_topic for v in picked.values()):
            break
    parts = []
    for topic in ORDER:
        if picked[topic]:
            items = "; ".join(f"{t} ({s})" if d == cutoff else f"{t} ({s}, {d})" for t, s, d in picked[topic])
            parts.append(f"{topic.capitalize() if topic != 'AI' else 'AI'}: {items}.")
    text = (f"Recent headlines from {cutoff}, for answering questions about recent news. Only mention them "
            f"when the user asks about news or current events. " + " ".join(parts))
    items = [{"title": t, "source": s} for topic in ORDER for t, s, _ in picked[topic]]
    Path(args.out).write_text(json.dumps({"cutoff": cutoff, "text": safe(text), "items": items}, ensure_ascii=False, indent=1))
    print(f"Digest: {sum(len(v) for v in picked.values())} headlines, news up to {cutoff}, {len(text)} chars -> {args.out}")


if __name__ == "__main__":
    main()
