#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "No Python executable found. Set PATH or activate the project environment." >&2
  exit 127
fi

echo "==> compileall"
"${PYTHON_CMD[@]}" -m compileall .

echo "==> reward dry-run"
"${PYTHON_CMD[@]}" -m task3_eval.rlfr.reward --dry_run --dummy_probe true

echo "==> train_grpo_rlfr reward-only dry-run"
"${PYTHON_CMD[@]}" -m task3_eval.rlfr.train_grpo_rlfr \
  --config configs/rlfr_pilot_lambda05.yaml \
  --dry_run_reward_only

echo "PASS"
