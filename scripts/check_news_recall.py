#!/usr/bin/env python3
"""
Measure how much of the trained news a GGUF build can actually recall.

Takes headlines the model was trained on (data/processed/distilled_news.jsonl),
asks about each one in different words than training used, and counts an answer
as a hit when it contains at least half of the summary's distinctive words that
the question itself doesn't give away. Prints the recall percentage.

Usage:
    python scripts/check_news_recall.py models/irx-1-Q4_K_M.gguf [--n 25]
"""

import argparse
import json
import random
import re
import subprocess
import time
import urllib.request

PORT = 8815
STOP = set("""about after again against their there these those which while would could should with from that this
have been were will into than then them they what when where your said says also over more most only other some such
very just like report reported news week year years month today first last year's""".split())


def words(text):
    return {w for w in re.findall(r"[a-z][a-z'\-]{3,}", text.lower()) if w not in STOP}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("--data", default="data/processed/distilled_news.jsonl")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--ngl", default="99")
    parser.add_argument("--digest", help="test the headlines in this digest JSON (answered from the built-in prompt)")
    args = parser.parse_args()

    items = []
    for line in open(args.data):
        m = json.loads(line)["messages"]
        q = m[0]["content"]
        if q.startswith("Tell me about this news: "):
            title = q.split(": ", 1)[1]
            summary = m[1]["content"].split(" (Reported by")[0]
            items.append((title, summary))
    random.Random(7).shuffle(items)
    items = items[: args.n]
    if args.digest:
        # Each digest headline: the answer must carry the headline's own facts.
        items = [(d["title"], d["title"]) for d in json.load(open(args.digest))["items"]]

    server = subprocess.Popen(["llama-server", "-m", args.gguf, "--jinja", "-c", "8192", "-ngl", args.ngl,
                               "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hits = 0
    try:
        for _ in range(120):
            try:
                if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2).read():
                    break
            except Exception:
                time.sleep(1)
        for title, summary in items:
            cue = " ".join(title.split()[:6]) if not args.digest else " ".join(title.split()[:4])
            question = f"What do you know about the recent news: {cue}?"
            body = {"messages": [{"role": "user", "content": question}], "temperature": 0.2, "max_tokens": 220, "seed": 1}
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            answer = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
            target = words(summary) - words(question)
            found = target & words(answer)
            ok = len(target) > 0 and len(found) >= max(2, len(target) // 2)
            hits += ok
            print(f"{'HIT ' if ok else 'miss'} {len(found)}/{len(target)}  {cue}")
    finally:
        server.terminate()
        server.wait()
    print(f"NEWS RECALL: {hits}/{len(items)} = {100 * hits / len(items):.0f}%")


if __name__ == "__main__":
    main()
