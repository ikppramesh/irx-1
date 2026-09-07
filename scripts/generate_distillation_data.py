#!/usr/bin/env python3
"""
Generate synthetic training examples by running a teacher model locally (via MLX)
over data/seed_prompts.txt. Output goes to data/processed/distilled_<tag>.jsonl,
in the same {"messages": [...]} format as prepare_data.py, ready for combine_data.py.

Usage:
    python scripts/generate_distillation_data.py --model mlx-community/Qwen3.5-2B-4bit --tag qwen
    python scripts/generate_distillation_data.py --model models/base/gemma-4-E2B-it-4bit --tag gemma
"""

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id or local MLX path")
    parser.add_argument("--tag", required=True, help="short label, e.g. qwen or gemma")
    parser.add_argument("--prompts", default="data/seed_prompts.txt")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    prompts = [
        line.strip()
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(prompts)} seed prompts")

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"distilled_{args.tag}.jsonl"

    system_prompt = (
        "Respond directly with only your final answer. Do not show your reasoning, "
        "planning, drafts, or a step-by-step thinking process."
    )

    with open(out_path, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            response = generate(
                model, tokenizer, prompt=formatted, max_tokens=args.max_tokens
            ).strip()

            # Defensive cleanup in case the model still emits a thinking block.
            if "</think>" in response:
                response = response.split("</think>", 1)[1].strip()
            if response.lower().startswith("thinking process"):
                continue

            if not response:
                continue

            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
            f.write(json.dumps(example) + "\n")
            print(f"[{i}/{len(prompts)}] {prompt[:60]!r} -> {len(response)} chars")

    print(f"Wrote distilled examples -> {out_path}")


if __name__ == "__main__":
    main()
