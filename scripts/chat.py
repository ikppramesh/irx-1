#!/usr/bin/env python3
"""
Interactive chat with IRx-1, with thinking-mode properly suppressed
(mlx_lm.chat doesn't expose this, and defaults to greedy decoding which
causes repetition loops on a lightly-trained LoRA model).

Usage:
    python scripts/chat.py
    python scripts/chat.py --model models/irx-1-merged --temp 0.7
"""

import argparse

from mlx_lm import stream_generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_SYSTEM_PROMPT = (
    "Respond directly with only your final answer. Do not show your reasoning, "
    "planning, drafts, or a step-by-step thinking process."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/irx-1-merged")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temp)
    print("Ready. Type your message and press enter. Type 'q' to quit, 'r' to reset.\n")

    history = []
    while True:
        try:
            query = input(">> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "q":
            break
        if query.strip().lower() == "r":
            history = []
            print("(history cleared)\n")
            continue
        if not query.strip():
            continue

        history.append({"role": "user", "content": query})
        messages = [{"role": "system", "content": args.system_prompt}] + history
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        response = ""
        for chunk in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler
        ):
            print(chunk.text, end="", flush=True)
            response += chunk.text
        print("\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
