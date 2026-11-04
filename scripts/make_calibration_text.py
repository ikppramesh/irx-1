#!/usr/bin/env python3
"""
Write llama-imatrix calibration text from a training split: the plain text of
500 shuffled conversations (~600K characters, comfortably over the 150 x 512
tokens llama-imatrix reads).

Usage:
    python scripts/make_calibration_text.py data/processed/train.jsonl out.txt
"""

import json
import random
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src) if line.strip()]
    random.Random(0).shuffle(rows)
    texts = [m["content"] for r in rows[:500] for m in r["messages"]]
    with open(out, "w") as f:
        f.write("\n\n".join(texts))


if __name__ == "__main__":
    main()
