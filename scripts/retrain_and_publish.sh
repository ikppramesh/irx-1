#!/usr/bin/env bash
# Weekly pipeline, Thursdays 11:15 IST (scripts/com.irx1.retrain.plist):
# fetch news -> news training data + digest -> rebuild all four IRx models
# (scripts/build_model.sh: train, GGUF, behaviour check, checkpoint fallback) ->
# publish each model that passes -> commit the news snapshot to GitHub.
#
# Everything heavy lives on the external drive ($IRX_HOME); if it isn't plugged in,
# the run is skipped (logged) and retried next week. One model failing doesn't stop
# the others; the run exits non-zero if any failed.
#
# History of scheduled-run failures this guards against:
#  - launchd's minimal PATH lacked Homebrew (llama-quantize: command not found).
#  - 2026-09: GPU out-of-memory at a 1024-token training window (now 768).
#  - 2026-10-29 and 2026-11-04: training froze the Mac into a kernel panic
#    (watchdog timeout) -- now capped with TRAIN_MEMORY_GB in scripts/finetune.sh,
#    and the Mac is kept awake for the run with caffeinate.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

IRX_HOME="${IRX_HOME:-/Volumes/1TB_IR/TrainedModels/IRx}"
LOG_DIR="$IRX_HOME/logs"
stamp() { date '+%F %T'; }

if [ ! -d "$IRX_HOME" ]; then
  echo "$(stamp) retrain skipped: drive not mounted ($IRX_HOME)" >> "$HOME/Library/Logs/irx-retrain.log"
  exit 0
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/retrain-$(date +%F).log"
exec >> "$LOG" 2>&1

# One run at a time (a missed run fires on wake and could overlap a manual one).
LOCK="$IRX_HOME/.retrain.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(stamp) another retrain is running ($LOCK) -- skipping"; exit 0
fi
trap 'rmdir "$LOCK"' EXIT
caffeinate -dims -w $$ &

source .venv/bin/activate
export HF_HOME="$IRX_HOME/hf_cache"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export TRAIN_MEMORY_GB="${TRAIN_MEMORY_GB:-11}"

echo "===== weekly retrain: $(stamp) ====="
echo "[1/4] Fetching latest news"
python scripts/fetch_news.py --json-out docs/news.json || echo "news fetch failed -- continuing with what's stored"

echo "[2/4] News training examples + digest"
python scripts/generate_news_training_data.py
python scripts/make_news_digest.py

echo "[3/4] Rebuilding models"
export HF_HUB_OFFLINE=1   # bases are cached on the drive; publishing below goes online
FAILED=""
for m in mini irx1 irx2 pro; do
  echo "----- $m: $(stamp) -----"
  if bash scripts/build_model.sh "$m" --publish > "$LOG_DIR/build-$m-$(date +%F).log" 2>&1; then
    echo "$m built, passed the check and published"
  else
    echo "$m FAILED -- see $LOG_DIR/build-$m-$(date +%F).log"; FAILED="$FAILED $m"
  fi
done

echo "[4/4] Committing the news snapshot"
git add docs/news.json
if ! git diff --staged --quiet; then
  git pull --rebase --autostash || true
  git commit -m "Weekly retrain: refresh news snapshot [skip ci]"
  git push || echo "git push failed -- will retry next week"
fi

echo "===== weekly retrain done: $(stamp) failed:${FAILED:- none} ====="
[ -z "$FAILED" ]
