#!/usr/bin/env python3
"""
Pre-publish behavior check for a GGUF build, run the way phone apps run it:
embedded chat template, no system prompt, the app's thinking request left on,
and no repeat penalty. Exits non-zero on any failure so a scheduled retrain
never publishes a regressed build.

Fails on:
  - hidden reasoning (the "conversation ran out of room" bug in PocketPal)
  - looping: hitting the token limit with repetitive text, or one sentence
    repeated 4+ times
  - multi-turn copying: an answer >80% identical to the previous one
  - identity: "who are you" answers missing the model name

Non-English prompts are reported but don't fail the check -- small bases are
weak there before any fine-tuning.

Usage:
    python scripts/check_gguf.py models/irx-1-Q4_K_M.gguf --name IRx-1
"""

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter

PORT = 8811
MAX_TOKENS = 1536

SINGLE = [
    "Plan a 3 day trip to Goa for a family of four with a moderate budget.",
    "Write a short essay about the importance of education.",
    "Explain machine learning to a 10 year old.",
    "What are the pros and cons of electric cars?",
    "Tell me about the history of Hyderabad.",
    "How do I start a small business in India?",
    "Describe the water cycle.",
    "Write a python function to check if a string is a palindrome.",
]
# Long day-by-day itineraries are where small models loop: IRx-mini (0.8B) repeated
# "Hyderabad to Hyderabad" on the first prompt in PocketPal (repeat penalty 1.0).
TRIPS = [
    "Plan a trip to Lakshadweep from Hyderabad and suggest me days as well to stay",
    "Plan a 5 day trip to Kerala from Bangalore with a day-by-day plan",
    "Give me a 7 day itinerary for Rajasthan",
]
CONVERSATION = [
    "Hi, I'm planning to learn cooking.", "What dishes should I start with?",
    "Tell me more about the first one.", "How long will it take to cook?",
    "Can you give me the full recipe step by step?", "What can go wrong?",
    "Thanks! Now suggest a dessert too.", "Summarize everything we discussed.",
]
IDENTITY = ["Who are you?", "Who created you?", "What is your name?", "Are you ChatGPT?", "Who trained you?"]
OTHER_MAKERS = ("qwen", "gemma", "alibaba", "openai", "google", "deepmind", "anthropic", "meta ai", "microsoft", "mistral")
INFO_ONLY = ["मुझे दिल्ली घूमने के लिए एक योजना बताइए।", "హైదరాబాద్ గురించి చెప్పండి."]


def ask(messages, seed):
    body = {"messages": messages, "temperature": 0.7, "top_p": 0.95, "min_p": 0.05,
            "repeat_penalty": 1.0, "max_tokens": MAX_TOKENS, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    m = r["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", r["choices"][0]["finish_reason"]


def loop_reason(text, reasoning, finish):
    if reasoning.strip():
        return f"hidden reasoning ({len(reasoning)} chars)"
    words = text.lower().split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 4)]
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    # A long, detailed answer can legitimately reach the cap; a loop is long
    # AND repetitive (the IRx-1 Q4_0/IQ4_NL loops measured 0.49 distinct).
    if finish == "length" and distinct < 0.6:
        return f"ran to the token limit repeating itself ({distinct:.0%} distinct 4-grams)"
    sentences = Counter(s.strip().lower() for s in re.split(r"(?<=[.!?\n])\s+", text) if len(s.strip()) > 15)
    if sentences and max(sentences.values()) >= 4:
        return "sentence repeated %dx" % max(sentences.values())
    if distinct < 0.45:
        return f"repetitive ({distinct:.0%} distinct 4-grams)"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ngl", default="99", help="GPU layers; 0 = CPU only")
    parser.add_argument("--chat-template-file", help="test a template other than the embedded one")
    args = parser.parse_args()

    cmd = ["llama-server", "-m", args.gguf, "--jinja", "-c", "16384", "-ngl", args.ngl, "--port", str(PORT)]
    if args.chat_template_file:
        cmd += ["--chat-template-file", args.chat_template_file]
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2).read():
                    break
            except Exception:
                time.sleep(1)
        failures = []

        for i, p in enumerate(SINGLE):
            text, reasoning, finish = ask([{"role": "user", "content": p}], seed=i)
            if reason := loop_reason(text, reasoning, finish):
                failures.append(f"[single] {p[:50]}: {reason} ...{text[-160:]!r}")

        for i, p in enumerate(TRIPS):
            for seed in range(3):
                text, reasoning, finish = ask([{"role": "user", "content": p}], seed=500 + 10 * i + seed)
                if reason := loop_reason(text, reasoning, finish):
                    failures.append(f"[trip seed {seed}] {p[:50]}: {reason} ...{text[-160:]!r}")

        # Loops are stochastic -- one pass can miss them, so run the chat 3 times.
        for run in range(3):
            messages, earlier = [], []
            for i, u in enumerate(CONVERSATION):
                messages.append({"role": "user", "content": u})
                text, reasoning, finish = ask(messages, seed=100 * (run + 1) + i)
                if reason := loop_reason(text, reasoning, finish):
                    failures.append(f"[run {run + 1} turn {i + 1}] {u}: {reason} ...{text[-120:]!r}")
                elif any(difflib.SequenceMatcher(None, text[:600], e[:600]).ratio() > 0.8 for e in earlier):
                    failures.append(f"[run {run + 1} turn {i + 1}] {u}: copied an earlier answer {text[:120]!r}")
                messages.append({"role": "assistant", "content": text})
                earlier.append(text)

        for i, q in enumerate(IDENTITY):
            text, reasoning, finish = ask([{"role": "user", "content": q}], seed=200 + i)
            low = text.lower()
            named = args.name.lower() in low or "ramesh" in low
            # "Are you ChatGPT?" -> "No, I'm not ChatGPT." is a correct answer.
            denied = q.startswith("Are you") and (low.startswith("no") or " not " in low)
            # A denial may name the product it isn't ("No, I'm not ChatGPT"); any other
            # answer must not credit another AI company (IRx-mini said "trained by OpenAI").
            # "No, I'm not made by OpenAI ... I'm built by Ramesh Inampudi" is also correct.
            claims = [w for w in OTHER_MAKERS if w in low and not (denied and (w in q.lower() or named))]
            if not (named or denied) or claims:
                failures.append(f"[identity] {q}: {text[:160]!r}")

        text, _, _ = ask([{"role": "user", "content": "How up to date is your knowledge?"}], seed=400)
        print(f"info (non-blocking) news cutoff answer: {text[:160]!r}")

        for i, p in enumerate(INFO_ONLY):
            text, reasoning, finish = ask([{"role": "user", "content": p}], seed=300 + i)
            print(f"info (non-blocking) {p[:20]}: {loop_reason(text, reasoning, finish) or 'ok'}")
    finally:
        server.terminate()
        server.wait()

    if failures:
        print(f"FAIL: {len(failures)} problem(s) in {args.gguf}")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"PASS: {args.gguf}")


if __name__ == "__main__":
    main()
