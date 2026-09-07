#!/usr/bin/env bash
# Serve IRx-1 as a local OpenAI-compatible HTTP API (for tools like XGAIR's
# chat intent parser to call). Runs entirely offline once the model is cached.
set -euo pipefail

MODEL="${1:-models/irx-1-merged}"
PORT="${IRX1_PORT:-8765}"

mlx_lm.server \
  --model "$MODEL" \
  --port "$PORT" \
  --temp 0.1 \
  --max-tokens 300 \
  --chat-template-args '{"enable_thinking":false}'
