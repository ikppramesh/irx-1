#!/usr/bin/env python3
"""
Generate trip-planning training examples via a local teacher model, same
distillation pattern as generate_distillation_data.py, but with a system
prompt that specifically shapes the OUTPUT FORMAT (day-by-day timeline,
cost table, food picks) -- not specific facts. This matters because of a
finding proven repeatedly in this project: fine-tuning reliably teaches
STYLE/STRUCTURE/BEHAVIOR (see the xGAIR intent-parsing data, the identity
attribution instruction) but does NOT reliably teach or update FACTS
(see news, geography/science limitations in the model card). Flight and
hotel prices are exactly the kind of fast-changing, precise fact
fine-tuning is worst at -- so the generation prompt below deliberately
forces cost figures into clearly-labeled estimate ranges with an explicit
"verify before booking" instruction baked into every example, rather than
teaching the model to state specific live prices with false confidence.

Usage:
    python scripts/generate_trip_planning_data.py
"""

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load

TRIP_SYSTEM_PROMPT = (
    "Respond directly with only your final answer. Do not show your reasoning, "
    "planning, drafts, or a step-by-step thinking process.\n\n"
    "You are helping plan a multi-day trip. Structure every answer exactly like this:\n"
    "1. Start with a one-line trip summary (route, duration, vibe).\n"
    "2. A day-by-day timeline: for each day, use a '### Day N' heading, then "
    "Morning / Afternoon / Evening bullet points covering where to go and what to do.\n"
    "3. Under each day, name 1-2 well-known local dishes or food spots worth trying.\n"
    "4. End with a markdown cost table with rows for local transport, activities/entry "
    "fees, food, and accommodation, each as an estimated per-day RANGE (e.g. '$15-25' or "
    "'₹800-1500'), never a single exact number, and never a specific flight or hotel "
    "booking price.\n"
    "5. Close with exactly this note, adapted to the currency used: 'These are typical "
    "estimates -- flight fares and hotel rates change constantly, so check a live search "
    "engine (e.g. Google Flights, Skyscanner, Booking.com) close to your travel dates "
    "before booking.'\n\n"
    "Use the local currency for the destination (₹ for India, $ for the US, local "
    "currency elsewhere) for the cost estimates. Never invent a specific numbered flight "
    "price, a specific hotel name with an exact nightly rate, or a specific restaurant "
    "bill total -- estimate ranges only."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    parser.add_argument("--tag", default="tripplanning")
    parser.add_argument("--prompts", default="data/trip_planning_prompts.txt")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-tokens", type=int, default=900)
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

    with open(out_path, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            messages = [
                {"role": "system", "content": TRIP_SYSTEM_PROMPT},
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

            if "</think>" in response:
                response = response.split("</think>", 1)[1].strip()
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

    print(f"Wrote trip-planning examples -> {out_path}")


if __name__ == "__main__":
    main()
