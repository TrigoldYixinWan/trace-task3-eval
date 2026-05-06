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

SMOKE_DIR="outputs/smoke_trace_probe4/TRACE_label_multiple_layer"
mkdir -p "$SMOKE_DIR"

"${PYTHON_CMD[@]}" -m compileall src/task3_eval/rlfr scripts

"${PYTHON_CMD[@]}" - <<'PY'
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

path = Path("outputs/smoke_trace_probe4/TRACE_label_multiple_layer/probe_4_layers.pk")
hidden_size = 2
num_layers = 4
dim = hidden_size * num_layers
X = np.asarray(
    [
        [0.0] * dim,
        [1.0] * dim,
        [0.1] * dim,
        [0.9] * dim,
    ],
    dtype=float,
)
y = np.asarray([0, 1, 0, 1], dtype=int)
model = LogisticRegression().fit(X, y)
joblib.dump({"models": {"activation_only": model}, "default_model": "activation_only"}, path)
PY

"${PYTHON_CMD[@]}" -m task3_eval.rlfr.probe_loader \
  --probe_path "$SMOKE_DIR/probe_4_layers.pk" \
  --probe_architecture sklearn \
  --layer_indices 1,2,3,4 \
  --pooling_method completion_mean_pool \
  --hidden_size 2 \
  --dry_run

echo "PASS"
