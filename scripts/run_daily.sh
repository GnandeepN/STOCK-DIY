#!/usr/bin/env bash
set -euo pipefail

# Resolve to repo root regardless of where it's called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

VENV="$REPO_DIR/venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Python venv not found at $PY" >&2
  exit 1
fi

# Run full pipeline; remove --force if you only want changes
"$PY" run_all.py --force --parallel

echo "[run_daily.sh] Completed at $(date)"

