#!/usr/bin/env python3
"""
Fix-ups for an MLX Gemma 3 build made with mlx_lm.fuse + mlx_lm.convert (IRx-mini).

  1. eos_token_id -> [1, 106]: the config only lists <eos>, so generation ran past
     <end_of_turn> into junk text.
  2. Restore the base's tokenizer files (tokenizer.model, special_tokens_map.json,
     added_tokens.json) and tokenizer_config.json without its chat_template (which
     would override chat_template.jinja); the re-saved config had lost the
     special-token table.
  3. Drop lm_head when it is an exact copy of the embedding table (Gemma ties them;
     mlx_lm re-ties when lm_head is absent) -- saves ~0.3GB.

Usage: python scripts/fix_mlx_gemma3.py <mlx dir> <base snapshot dir>
"""

import json
import os
import shutil
import sys

import mlx.core as mx


def main():
    out, base = sys.argv[1], sys.argv[2]
    cfg = json.load(open(f"{out}/config.json"))
    cfg["eos_token_id"] = [1, 106]
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=4)

    tok = json.load(open(f"{base}/tokenizer_config.json"))
    tok.pop("chat_template", None)
    json.dump(tok, open(f"{out}/tokenizer_config.json", "w"), indent=2)
    for f in ("tokenizer.model", "special_tokens_map.json", "added_tokens.json"):
        if os.path.exists(f"{base}/{f}"):
            shutil.copy(f"{base}/{f}", out)

    path = f"{out}/model.safetensors"
    w = mx.load(path)
    parts = [k for k in ("weight", "scales", "biases") if f"lm_head.{k}" in w]
    if parts and all(f"model.embed_tokens.{k}" in w and
                     mx.array_equal(w[f"lm_head.{k}"], w[f"model.embed_tokens.{k}"]).item() for k in parts):
        for k in parts:
            del w[f"lm_head.{k}"]
        mx.eval(w)
        tmp = f"{out}/model.tmp.safetensors"
        mx.save_safetensors(tmp, w, metadata={"format": "mlx"})
        os.replace(tmp, path)
        index = f"{out}/model.safetensors.index.json"
        if os.path.exists(index):
            j = json.load(open(index))
            j["weight_map"] = {k: v for k, v in j["weight_map"].items() if not k.startswith("lm_head")}
            json.dump(j, open(index, "w"), indent=2)
        print("Dropped tied lm_head")
    print(f"Fixed {out}")


if __name__ == "__main__":
    main()
