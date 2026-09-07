#!/usr/bin/env python3
"""
Merge data/processed/personal.jsonl and data/processed/distilled_*.jsonl into the
final MLX-LM training split: data/processed/train.jsonl, data/processed/valid.jsonl.

Usage:
    python scripts/combine_data.py
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sources = sorted(data_dir.glob("personal.jsonl")) + sorted(
        data_dir.glob("distilled_*.jsonl")
    )
    if not sources:
        raise SystemExit(
            f"No source files found in {data_dir}. Run prepare_data.py and/or "
            "generate_distillation_data.py first."
        )

    examples = []
    for src in sources:
        with open(src) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        print(f"{src.name}: {len(lines)} examples")
        examples.extend(lines)

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val, train = examples[:n_val], examples[n_val:]

    for name, split in [("train.jsonl", train), ("valid.jsonl", val)]:
        out_path = data_dir / name
        with open(out_path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split)} examples -> {out_path}")


if __name__ == "__main__":
    main()
