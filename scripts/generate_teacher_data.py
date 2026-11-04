#!/usr/bin/env python3
"""
Distil a stronger local teacher into IRx training data, for reasoning and friendly
everyday conversation.

Reads data/teacher_prompts.jsonl (scripts/build_teacher_prompts.py) and writes
data/processed/distilled_teacher.jsonl, which combine_data.py picks up with the
other distilled_*.jsonl sources.

  reasoning  -- the teacher solves the maths problem in short, visible steps and ends
                with "**Answer:** <number>". Kept only if that number matches the
                known answer (rejection sampling), so the students never learn a
                wrong method. Visible steps rather than hidden thinking: IRx keeps
                thinking mode off on phones (it ran out the context in PocketPal).
  chat /
  everyday   -- answered warmly and accurately, sized to the question.

Batched generation (mlx_lm.batch_generate) and a memory cap keep it fast without
starving the rest of the Mac (an uncapped run froze it into a kernel panic).
Resumable: prompts already in the output file are skipped.

Usage:
    python scripts/generate_teacher_data.py [--model mlx-community/Qwen3.5-9B-4bit] [--batch 8]
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

STYLE = (
    "You are a warm, friendly and accurate assistant helping everyday people -- students, "
    "farmers, shop owners, parents, developers, travellers. Be kind and encouraging, use "
    "simple clear language, and get the facts right. Match the length to the question: "
    "short for simple questions, well-structured (short paragraphs, lists or steps) for "
    "bigger ones. If something is uncertain or risky (health, money, law), say so gently "
    "and suggest a professional where it matters. Never repeat yourself. Do not mention "
    "being an AI model from any company. Write maths in plain text that reads well in a "
    "phone chat app (for example: 15 × 0.2 = 3), never LaTeX or $...$."
)
REASONING = (
    STYLE + " For this problem, reason step by step: number each step, show the "
    "calculation in each, keep the steps short, and finish with a final line in exactly "
    "this form: **Answer:** <number>"
)
ANSWER = re.compile(r"\*\*Answer:\*\*\s*\$?\s*([-\d,]*\.?\d+)")


def plain_math(text):
    """LaTeX -> plain text: phone chat apps show $15 \\times 0.2$ as raw symbols.
    The style prompt forbids LaTeX, but ~7% of answers used it anyway. Money ("$15")
    is left alone: only $...$ spans that contain a LaTeX command are unwrapped."""
    text = re.sub(r"\$([^$\n]*\\[a-zA-Z]+[^$\n]*)\$", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", text)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\times", "×"), ("\\cdot", "·"),
                 ("\\div", "÷"), ("\\approx", "≈"), ("\\leq", "≤"), ("\\geq", "≥"),
                 ("\\%", "%"), ("\\$", "$"), ("\\,", " "), ("\\(", ""), ("\\)", ""),
                 ("\\[", ""), ("\\]", "")):
        text = text.replace(a, b)
    text = re.sub(r"(\d)\$(?=[\s.,;:)]|$)", r"\1", text)   # stray closing $ after a number
    return re.sub(r"[ \t]{2,}", " ", text)


def number(s):
    try:
        v = float(s.replace(",", ""))
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    parser.add_argument("--prompts", default="data/teacher_prompts.jsonl")
    parser.add_argument("--out", default="data/processed/distilled_teacher.jsonl")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=11.0)
    parser.add_argument("--clean-only", action="store_true", help="re-clean an existing output file and exit")
    args = parser.parse_args()

    if args.clean_only:
        rows = [json.loads(l) for l in open(args.out)]
        for r in rows:
            r["messages"][-1]["content"] = plain_math(r["messages"][-1]["content"])
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"Cleaned {len(rows)} examples in {args.out}")
        return

    mx.set_memory_limit(int(args.memory_gb * 2**30))
    mx.set_wired_limit(int(args.memory_gb * 0.8 * 2**30))

    rows = [json.loads(l) for l in open(args.prompts)]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["messages"][0]["content"] for l in open(out)}
    todo = [r for r in rows if r["prompt"] not in done]
    print(f"{len(rows)} prompts, {len(done)} already done, {len(todo)} to go")

    model, tok = load(args.model)
    sampler = make_sampler(temp=0.6, top_p=0.95)
    kept = dropped = 0
    with open(out, "a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i: i + args.batch]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": REASONING if r["kind"] == "reasoning" else STYLE},
                 {"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True, enable_thinking=False) for r in chunk]
            max_tokens = [600 if r["kind"] == "reasoning" else 900 for r in chunk]
            texts = batch_generate(model, tok, prompts, max_tokens=max_tokens, sampler=sampler).texts
            for r, text in zip(chunk, texts):
                text = plain_math(re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip())
                if r["kind"] == "reasoning":
                    m = ANSWER.search(text)
                    if not m or number(m.group(1)) != number(r["answer"]):
                        dropped += 1
                        continue
                if len(text) < 2 or text.count("\n\n\n\n") > 2:
                    dropped += 1
                    continue
                f.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                                 {"role": "assistant", "content": text}]},
                                   ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}  kept {kept}  dropped {dropped}", flush=True)
    print(f"Done: kept {kept}, dropped {dropped} -> {out}")


if __name__ == "__main__":
    main()
