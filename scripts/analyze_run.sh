#!/usr/bin/env bash
# Wrapper for scripts/analyze_run.py that picks an interpreter with tensorboard installed.
# Run from the repo root:  ./scripts/analyze_run.sh [--baseline <run>] [--run <run>] [--at <step>]
set -euo pipefail
cd "$(dirname "$0")/.."
for py in tensor-env/bin/python ml_env/bin/python python3; do
  if [ -x "$py" ] || command -v "$py" >/dev/null 2>&1; then
    if "$py" -c "import tensorboard" >/dev/null 2>&1; then
      exec env TF_CPP_MIN_LOG_LEVEL=3 "$py" scripts/analyze_run.py "$@" 2>/dev/null
    fi
  fi
done
echo "No interpreter with tensorboard found (tried tensor-env, ml_env, python3)." >&2
exit 1
