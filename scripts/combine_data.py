#!/usr/bin/env python3
"""
Merge data/processed/personal.jsonl, identity.jsonl and distilled_*.jsonl into
the final MLX-LM training split: data/processed/train.jsonl, valid.jsonl.

Every example is fitted to --max-tokens (keep in sync with finetune.sh's
--max-seq-length). mlx_lm otherwise silently truncates long examples, which
cuts the final assistant answer mid-sentence -- teaching the model to stop
early -- and long sequences are what pushed training into Metal OOM. Multi-turn
conversations drop their oldest exchanges first; anything still too long
(a single answer longer than the window) is dropped.

Usage:
    python scripts/combine_data.py
"""

import argparse
import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from make_chat_template import PROMPTS

BASE_MODEL = "mlx-community/Qwen3.5-2B-4bit"


# Chat-export UI text that leaked into assistant turns (17 occurrences).
BOILERPLATE_LINES = {
    "Viewing artifacts created via the Analysis Tool web feature preview isn’t yet supported on mobile.",
}
REFUSAL = re.compile(r"\b(can.?t|cannot) (assist|help) with (that|this)\b|^I'm sorry, but", re.I)


def strip_boilerplate(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant":
            lines = [ln for ln in m["content"].split("\n") if ln.strip() not in BOILERPLATE_LINES]
            m = {**m, "content": "\n".join(lines).strip()}
        out.append(m)
    return out


def reject_reason(messages):
    """Final answers that teach looping or refusing -- the failure modes seen on phones."""
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return "no final answer"
    final = messages[-1]["content"]
    if not final.strip():
        return "empty"
    words = final.lower().split()
    grams = Counter(tuple(words[i:i + 6]) for i in range(len(words) - 6))
    if grams and max(grams.values()) >= 4:
        return "repetitive"
    if REFUSAL.search(final):
        return "refusal"
    earlier = [m["content"] for m in messages[:-1] if m["role"] == "assistant"]
    if any(difflib.SequenceMatcher(None, final[:600], e[:600]).ratio() > 0.8 for e in earlier):
        return "copies an earlier answer"
    return None


def normalize_turns(messages):
    """Merge back-to-back turns from the same role and drop a leading assistant turn.
    Some personal chats have them; Gemma's chat template rejects non-alternating roles."""
    head = messages[:1] if messages[0]["role"] == "system" else []
    out = []
    for m in messages[len(head):]:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        elif out or m["role"] == "user":
            out.append(dict(m))
    return head + out


def token_len(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text)["input_ids"])


def fit(tokenizer, messages, max_tokens):
    """Drop oldest user/assistant exchanges until the example fits, or None."""
    head = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(head):]
    while token_len(tokenizer, head + body) > max_tokens:
        # Next exchange must still start with a user turn.
        nxt = next((i for i in range(1, len(body)) if body[i]["role"] == "user"), None)
        if nxt is None:
            return None
        body = body[nxt:]
    return head + body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default=BASE_MODEL, help="tokenizer used to measure lengths")
    parser.add_argument("--name", default="IRx-1", help="model name in the default system prompt")
    parser.add_argument("--prompt-style", choices=sorted(PROMPTS), default="default",
                        help="must match the chat template's (scripts/make_chat_template.py)")
    parser.add_argument("--system-prompt-fraction", type=float, default=0.6,
                        help="share of examples without a system prompt that get the default one")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sources = (
        sorted(data_dir.glob("personal.jsonl"))
        + sorted(data_dir.glob("identity.jsonl"))
        + sorted(data_dir.glob("distilled_*.jsonl"))
    )
    if not sources:
        raise SystemExit(
            f"No source files found in {data_dir}. Run prepare_data.py and/or "
            "generate_distillation_data.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # The chat template injects the default identity system prompt whenever an
    # app sends none -- i.e. in most real use. If that prompt only ever appears
    # in training on "who are you" examples, the model learns "this prompt ->
    # introduce yourself" and (seen in IRx-2) answers every turn with its
    # introduction. Put it on ordinary examples too, so training matches use.
    # Added before fitting, so the ~100 extra tokens count toward the window.
    default_prompt = PROMPTS[args.prompt_style].format(name=args.name)
    sp_rng = random.Random(args.seed + 1)

    examples = []
    for src in sources:
        with open(src) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        kept = trimmed = dropped = 0
        rejected = Counter()
        for ex in lines:
            messages = normalize_turns(strip_boilerplate(ex["messages"]))
            reason = reject_reason(messages)
            if reason:
                rejected[reason] += 1
                continue
            if messages[0]["role"] != "system" and sp_rng.random() < args.system_prompt_fraction:
                messages = [{"role": "system", "content": default_prompt}] + messages
            fitted = fit(tokenizer, messages, args.max_tokens)
            if fitted is None:
                dropped += 1
                continue
            trimmed += len(fitted) != len(messages)
            kept += 1
            examples.append({"messages": fitted})
        extra = "".join(f", {n} {r}" for r, n in rejected.items())
        print(f"{src.name}: {len(lines)} examples ({kept} kept, {trimmed} history-trimmed, {dropped} too long{extra})")

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
