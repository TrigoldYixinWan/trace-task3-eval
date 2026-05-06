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

echo "==> sklearn pkl probe-loader smoke"
SMOKE_PROBE_DIR="outputs/smoke_rlfr_reward/label_best_layer"
mkdir -p "$SMOKE_PROBE_DIR"
"${PYTHON_CMD[@]}" - <<'PY'
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

path = Path("outputs/smoke_rlfr_reward/label_best_layer/probe_model.pkl")
X = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]], dtype=float)
y = np.asarray([0, 1, 0, 1], dtype=int)
model = LogisticRegression().fit(X, y)
joblib.dump({"models": {"activation_only": model}, "default_model": "activation_only"}, path)
PY
"${PYTHON_CMD[@]}" -m task3_eval.rlfr.probe_loader \
  --probe_path "$SMOKE_PROBE_DIR" \
  --probe_model_key activation_only \
  --layer_idx 8 \
  --pooling_method completion_mean_pool \
  --hidden_size 2 \
  --dry_run

echo "==> train_grpo_rlfr reward-only dry-run"
"${PYTHON_CMD[@]}" -m task3_eval.rlfr.train_grpo_rlfr \
  --config configs/rlfr_pilot_lambda05.yaml \
  --dry_run_reward_only

echo "PASS"
