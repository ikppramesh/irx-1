#!/usr/bin/env python3
"""
Parse a claude.ai data export (data/raw/conversations.json) into MLX-LM
LoRA training format (data/processed/personal.jsonl). Run scripts/combine_data.py
afterward to merge with distillation data and produce the final train/valid split.

Usage:
    python scripts/prepare_data.py --input data/raw/conversations.json
"""

import argparse
import json
import random
import re
from pathlib import Path

# Redact obvious secrets/credentials/PII before anything touches a trained model.
# Best-effort, not a guarantee — spot-check data/processed/personal.jsonl by hand
# before it goes anywhere near training.
REDACT_PATTERNS = [
    # PEM private key / certificate blocks
    (re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----", re.S), "[PRIVATE_KEY]"),

    # "password: xxx", "api_key = xxx", "the password is xxx", etc. — keeps the
    # keyword, drops the value. Runs before the narrower patterns below.
    (re.compile(
        r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?key|"
        r"private[_-]?key|auth[_-]?token|credential(?:s)?)\b\s*(?:is|:|=)\s*"
        r"[\"']?[^\s\"'\n]+[\"']?"
    ), r"\1: [REDACTED]"),

    # connection strings with embedded credentials, e.g. postgres://user:pass@host
    (re.compile(r"\b([a-zA-Z][\w+.-]*)://[^\s/:@]+:[^\s/:@]+@"), r"\1://[REDACTED]@"),

    # JWTs (three dot-separated base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[JWT]"),

    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}\b", re.I), "[BEARER_TOKEN]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "[API_KEY]"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[AWS_KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[GITHUB_TOKEN]"),
    (re.compile(r"\b\d{1,5}\s+\w+(?:\s\w+){0,3}\s(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b", re.I), "[ADDRESS]"),

    # the account owner's own name/handle — kept out of training data by request,
    # since IRx-1 will be publicly queryable once released
    (re.compile(r"(?i)ikppramesh"), "[NAME]"),
    (re.compile(r"(?i)inampudi"), "[NAME]"),
    (re.compile(r"(?i)ramesh"), "[NAME]"),
]

MIN_CHARS = 4          # skip near-empty turns
MAX_CHARS = 4000        # skip pathologically long turns (code dumps, pastes)
MAX_HISTORY_MESSAGES = 6  # cap context window per example (3 user/assistant turns) —
                           # unbounded history in long conversations produced
                           # 9000+ token examples that OOM'd the GPU during training

# Conversations excluded by manual review (2026-09-07) — regex can't reliably
# catch full names in arbitrary scripts, birth details, or medical/financial
# specifics embedded in prose, so these are dropped wholesale rather than
# redacted in place. See conversation titles in data/raw/_conversation_titles.txt.
EXCLUDED_CONVERSATION_INDICES = {
    39, 40, 202,        # resume/CV: full name, contact info, work history
    42, 43, 44,         # astrology/birth-chart: full names, DOB/TOB/POB
    1, 55, 89, 108, 139,  # child's name (Aadya) and schedule/activities
    79, 80,             # Claude account changes mentioning a stored credit card
    94, 114, 115, 153,  # personal/family health & medical details
    143, 162,           # property/land legal documents with owner/ID details
    148,                # invitation card: child's full name + venue address
    157,                # child's name meaning
    172,                # contains actual generated example passwords
}

# Cleanup for a claude.ai export artifact: tool-use/artifact blocks that didn't
# survive as text get replaced with this placeholder. Left in, a fine-tuned
# model would learn to output it as if it were real assistant content.
TOOL_PLACEHOLDER_PATTERNS = [
    re.compile(r"```\s*\nThis block is not supported on your current device yet\.\s*\n```"),
    re.compile(r"This block is not supported on your current device yet\."),
]


def strip_tool_placeholders(text: str) -> str:
    for pattern in TOOL_PLACEHOLDER_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def redact(text: str) -> str:
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def extract_pairs(conversations: list[dict]) -> list[dict]:
    """Turn each conversation into one or more {messages: [...]} examples,
    one example per human turn using all prior context as history."""
    examples = []
    for i, convo in enumerate(conversations):
        if i in EXCLUDED_CONVERSATION_INDICES:
            continue
        messages = convo.get("chat_messages", [])
        history = []
        for msg in messages:
            sender = msg.get("sender")
            text = strip_tool_placeholders(msg.get("text") or "")
            if not text:
                continue
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                history = []  # bail on this thread if a turn is degenerate
                continue
            role = "user" if sender == "human" else "assistant"
            history.append({"role": role, "content": redact(text)})

            # Every time we complete a user->assistant pair, emit an example
            # using a capped window of recent turns as context (not the whole
            # conversation, which grows unbounded in long threads).
            if role == "assistant" and len(history) >= 2 and history[-2]["role"] == "user":
                window = history[-MAX_HISTORY_MESSAGES:]
                if window[0]["role"] != "user":
                    window = window[1:]
                examples.append({"messages": window})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/conversations.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Export your data from claude.ai "
            "(Settings -> Account -> Export data) and place conversations.json there."
        )

    with open(src) as f:
        conversations = json.load(f)

    examples = extract_pairs(conversations)
    print(f"Extracted {len(examples)} training examples from {len(conversations)} conversations")

    random.Random(args.seed).shuffle(examples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personal.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples -> {out_path}")
    print("Next: run scripts/generate_distillation_data.py, then scripts/combine_data.py")


if __name__ == "__main__":
    main()
