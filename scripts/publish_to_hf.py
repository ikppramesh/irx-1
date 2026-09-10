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
# The local build filename (models/irx-1-qwen3.5-2b-Q4_K_M.gguf) has never
# matched this -- always upload under the repo's one canonical filename so
# repeated automated runs overwrite in place instead of accumulating
# same-content duplicates under different names.
GGUF_REPO_FILENAME = "irx-1-Q4_K_M.gguf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", default="models/irx-1-merged")
    parser.add_argument("--gguf-file", default="models/irx-1-qwen3.5-2b-Q4_K_M.gguf")
    parser.add_argument("--gguf-readme", default="models/GGUF_README.md")
    parser.add_argument("--commit-message", default="Automated retrain: refresh weights")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.skip_mlx:
        merged_dir = Path(args.merged_dir)
        if not merged_dir.exists():
            raise SystemExit(f"Merged model dir not found: {merged_dir}")
        print(f"Uploading {merged_dir} -> {MODEL_REPO}")
        api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(merged_dir),
            commit_message=args.commit_message,
        )

    if not args.skip_gguf:
        gguf_path = Path(args.gguf_file)
        if not gguf_path.exists():
            raise SystemExit(f"GGUF file not found: {gguf_path}")
        print(f"Uploading {gguf_path.name} -> {GGUF_REPO} as {GGUF_REPO_FILENAME}")
        api.upload_file(
            repo_id=GGUF_REPO,
            path_or_fileobj=str(gguf_path),
            path_in_repo=GGUF_REPO_FILENAME,
            commit_message=args.commit_message,
        )
        readme_path = Path(args.gguf_readme)
        if readme_path.exists():
            api.upload_file(
                repo_id=GGUF_REPO,
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                commit_message=args.commit_message,
            )

        # Small manifest apps can poll instead of re-downloading the whole
        # GGUF just to check whether a new build exists. Written after the
        # GGUF itself so it never points at a version that isn't live yet.
        print("Publishing version manifest")
        sha256 = hashlib.sha256()
        with open(gguf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha256.update(chunk)
        manifest = {
            "version": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "filename": GGUF_REPO_FILENAME,
            "size_bytes": gguf_path.stat().st_size,
            "sha256": sha256.hexdigest(),
        }
        api.upload_file(
            repo_id=GGUF_REPO,
            path_or_fileobj=json.dumps(manifest, indent=2).encode(),
            path_in_repo="version.json",
            commit_message=args.commit_message,
        )

    print("Publish complete.")


if __name__ == "__main__":
    main()
