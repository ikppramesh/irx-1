#!/usr/bin/env python3
"""
Push the merged MLX model -> ikppramesh/irx-1 and the GGUF build ->
ikppramesh/irx-1-GGUF. Called by scripts/retrain_and_publish.sh after a
fresh mlx_lm.fuse + merge_and_quantize.sh run. huggingface_hub only
re-uploads files whose content actually changed, so README/diagram assets
already on the Hub are left untouched.

Usage:
    python scripts/publish_to_hf.py
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi

MODEL_REPO = "ikppramesh/irx-1"
GGUF_REPO = "ikppramesh/irx-1-GGUF"
# Local build filenames (models/irx-1-Q4_K_M.gguf) never match the
# repo's -- always upload under one canonical <prefix>-<quant>.gguf name so
# repeated automated runs overwrite in place instead of accumulating
# same-content duplicates under different names.


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--gguf-repo", default=GGUF_REPO)
    parser.add_argument("--gguf-prefix", default="irx-1", help="repo filenames: <prefix>-<quant>.gguf")
    parser.add_argument("--gguf-quant", default="Q4_K_M")
    parser.add_argument("--merged-dir", default="models/irx-1-merged")
    parser.add_argument("--gguf-file", default="models/irx-1-Q4_K_M.gguf")
    parser.add_argument("--gguf-q4-0-file", default="models/irx-1-Q4_0.gguf")
    parser.add_argument("--gguf-readme", default="models/GGUF_README.md")
    parser.add_argument("--commit-message", default="Automated retrain: refresh weights")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    model_repo, gguf_repo = args.model_repo, args.gguf_repo
    gguf_name = f"{args.gguf_prefix}-{args.gguf_quant}.gguf"
    gguf_q4_0_name = f"{args.gguf_prefix}-Q4_0.gguf"

    if not args.skip_mlx:
        merged_dir = Path(args.merged_dir)
        if not merged_dir.exists():
            raise SystemExit(f"Merged model dir not found: {merged_dir}")
        print(f"Uploading {merged_dir} -> {model_repo}")
        api.create_repo(model_repo, exist_ok=True)
        api.upload_folder(
            repo_id=model_repo,
            folder_path=str(merged_dir),
            commit_message=args.commit_message,
        )

    if not args.skip_gguf:
        gguf_path = Path(args.gguf_file)
        if not gguf_path.exists():
            raise SystemExit(f"GGUF file not found: {gguf_path}")
        print(f"Uploading {gguf_path.name} -> {gguf_repo} as {gguf_name}")
        api.create_repo(gguf_repo, exist_ok=True)
        api.upload_file(
            repo_id=gguf_repo,
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_name,
            commit_message=args.commit_message,
        )
        q4_0_path = Path(args.gguf_q4_0_file)
        if q4_0_path.exists():
            print(f"Uploading {q4_0_path.name} -> {gguf_repo} as {gguf_q4_0_name}")
            api.upload_file(
                repo_id=gguf_repo,
                path_or_fileobj=str(q4_0_path),
                path_in_repo=gguf_q4_0_name,
                commit_message=args.commit_message,
            )
        readme_path = Path(args.gguf_readme)
        if readme_path.exists():
            api.upload_file(
                repo_id=gguf_repo,
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                commit_message=args.commit_message,
            )

        # Small manifest apps can poll instead of re-downloading the whole
        # GGUF just to check whether a new build exists. Written after the
        # GGUF itself so it never points at a version that isn't live yet.
        print("Publishing version manifest")
        # Top-level keys unchanged (existing apps read them); "variants" is additive.
        manifest = {
            "version": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "filename": gguf_name,
            "size_bytes": gguf_path.stat().st_size,
            "sha256": sha256_of(gguf_path),
        }
        if q4_0_path.exists():
            manifest["variants"] = [
                {"filename": gguf_name, "size_bytes": manifest["size_bytes"],
                 "sha256": manifest["sha256"], "recommended_for": "iPhone, iPad, Mac (Metal GPU)"},
                {"filename": gguf_q4_0_name, "size_bytes": q4_0_path.stat().st_size,
                 "sha256": sha256_of(q4_0_path), "recommended_for": "Android (ARM CPU)"},
            ]
        api.upload_file(
            repo_id=gguf_repo,
            path_or_fileobj=json.dumps(manifest, indent=2).encode(),
            path_in_repo="version.json",
            commit_message=args.commit_message,
        )

    print("Publish complete.")


if __name__ == "__main__":
    main()
