#!/usr/bin/env python3
"""
Write the Hugging Face model cards from the templates in scripts/cards/.

  scripts/cards/<id>.md       -> models/<id>-merged/README.md   (MLX repo)
  scripts/cards/<id>-GGUF.md  -> models/cards/<id>-GGUF.md      (GGUF repo)

Fills {{NEWS_DATE}} (the digest's cutoff, data/news_digest.json) and {{UPDATED}}
(today). Edit the templates for anything else -- changelog entries included.

Usage: python scripts/make_model_cards.py
"""

import datetime
import json
from pathlib import Path

IDS = ("irx-1", "irx-2", "irx-2-pro", "irx-mini")


def main():
    cutoff = json.load(open("data/news_digest.json"))["cutoff"]
    today = datetime.date.today()
    updated = f"{today.day} {today:%B %Y}"
    Path("models/cards").mkdir(parents=True, exist_ok=True)
    for i in IDS:
        for src, dst in ((f"scripts/cards/{i}.md", f"models/{i}-merged/README.md"),
                         (f"scripts/cards/{i}-GGUF.md", f"models/cards/{i}-GGUF.md")):
            if not Path(dst).parent.exists():
                continue
            text = Path(src).read_text().replace("{{NEWS_DATE}}", cutoff).replace("{{UPDATED}}", updated)
            Path(dst).write_text(text)
    print(f"Cards written (news up to {cutoff}, updated {updated})")


if __name__ == "__main__":
    main()
