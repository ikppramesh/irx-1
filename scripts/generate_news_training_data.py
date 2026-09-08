#!/usr/bin/env python3
"""
Turn the most recently fetched articles in data/news.db into a small, bounded
set of training examples (data/processed/distilled_news.jsonl), so the
recurring retrain (scripts/retrain_and_publish.sh, triggered after every
news-fetch cron cycle) has something new to train on each time.

Deliberately template-based, not teacher-generated: the goal here is a
faithful Q/A pair grounded directly in the fetched title/summary, not a
paraphrase that could drift or hallucinate on top of already-terse RSS text.

Known limitation (documented in the model card, tested directly): fine-tuning
does not reliably teach specific, fast-changing facts the way retrieval does
-- the model's actual current-events grounding still comes from
scripts/news_context.py / src/utils/news.ts at answer-time, not from this
file. This just gives the model repeated exposure to "recent news" phrasing
and structure. This file is OVERWRITTEN (not appended) on every run, so the
training mix always reflects the latest cycle's articles rather than growing
unbounded across months of cron runs.

Usage:
    python scripts/generate_news_training_data.py
"""

import argparse
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/news.db")
OUT_PATH = Path("data/processed/distilled_news.jsonl")
MAX_ARTICLES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--max-articles", type=int, default=MAX_ARTICLES)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No news index at {args.db} -- run scripts/fetch_news.py first. Skipping.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")  # keep the rolling file empty rather than stale
        return

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT source, title, summary FROM articles ORDER BY fetched_at DESC LIMIT ?",
        (args.max_articles,),
    ).fetchall()
    conn.close()

    examples = []
    for source, title, summary in rows:
        summary = (summary or "").strip()[:400]
        if not title.strip():
            continue

        examples.append({
            "messages": [
                {"role": "user", "content": f"What's the latest news from {source}?"},
                {"role": "assistant", "content": f"{title.strip()}. {summary}".strip()},
            ]
        })
        if summary:
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Summarize this: {title.strip()}"},
                    {"role": "assistant", "content": f"{summary} (Source: {source})"},
                ]
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples from {len(rows)} articles -> {args.out}")


if __name__ == "__main__":
    main()
