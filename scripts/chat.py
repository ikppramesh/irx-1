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

from news_context import get_relevant_context

DEFAULT_SYSTEM_PROMPT = (
    "Respond directly with only your final answer. Do not show your reasoning, "
    "planning, drafts, or a step-by-step thinking process. "
    "Your name is IRx-1. If asked who you are, what you are, who created/made/built "
    "you, who your developer or author is, or anything about the identity or "
    "background of this model, always answer in your own words that you are IRx-1, "
    "created by Ramesh Inampudi from Hyderabad, India, and point to his website "
    "iramesh.com. Never mention any other AI company or base model name."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/irx-1-merged")
    parser.add_argument("--temp", type=float, default=0.7)
    # mlx_lm has no "unlimited" sentinel (unlike llama.cpp's n_predict=-1) --
    # generation always stops early via EOS in normal use, so this is a
    # generous backstop against a genuine runaway/repetition loop, not a
    # creativity-limiting cap like the old 512/1536 defaults were.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-news", action="store_true",
                         help="Disable news-index retrieval context")
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

        system_prompt = args.system_prompt
        if not args.no_news:
            context = get_relevant_context(query)
            if context:
                system_prompt = f"{system_prompt}\n\n{context}"

        messages = [{"role": "system", "content": system_prompt}] + history
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
