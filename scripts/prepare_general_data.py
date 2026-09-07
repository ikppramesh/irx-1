#!/usr/bin/env python3
"""
Sample a balanced subset of databricks-dolly-15k (CC-BY-SA-3.0, human-written
instruction/response pairs) to broaden IRx-1's general Q&A competence beyond
personal style + the earlier small distillation set.

Usage:
    python scripts/prepare_general_data.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

MAX_CHARS = 4000  # match prepare_data.py's per-turn cap
PER_CATEGORY = 50  # balanced sample across Dolly's 8 categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = hf_hub_download(
        "databricks/databricks-dolly-15k",
        "databricks-dolly-15k.jsonl",
        repo_type="dataset",
    )
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    by_category = defaultdict(list)
    for row in rows:
        instruction = row["instruction"].strip()
        context = row["context"].strip()
        response = row["response"].strip()
        if not instruction or not response:
            continue
        prompt = f"{instruction}\n\n{context}" if context else instruction
        if len(prompt) > MAX_CHARS or len(response) > MAX_CHARS:
            continue
        by_category[row["category"]].append(
            {"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]}
        )

    rng = random.Random(args.seed)
    sample = []
    for category, examples in by_category.items():
        rng.shuffle(examples)
        picked = examples[: args.per_category]
        sample.extend(picked)
        print(f"{category}: {len(picked)}/{len(examples)} sampled")

    rng.shuffle(sample)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "distilled_dolly.jsonl"
    with open(out_path, "w") as f:
        for ex in sample:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(sample)} examples -> {out_path}")


if __name__ == "__main__":
    main()
