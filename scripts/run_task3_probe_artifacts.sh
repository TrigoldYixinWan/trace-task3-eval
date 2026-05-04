#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-base}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
RUN_TYPE="${RUN_TYPE:-hacking}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
LIMIT="${LIMIT:-}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
EXTRACT_FEATURES="${EXTRACT_FEATURES:-0}"
EXTRACT_ALL_TOKEN="${EXTRACT_ALL_TOKEN:-0}"
PROBE_DATASET_PATH="${PROBE_DATASET_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "No Python executable found. Set PYTHON_BIN=/path/to/python." >&2
  exit 127
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_task3_probe_artifacts.sh SCORED_ROLLOUT.jsonl [MORE_SCORED_ROLLOUTS...]" >&2
  echo "Set EXTRACT_FEATURES=1 plus CHECKPOINT_PATH=/path/to/checkpoint to create hidden-state features and train a probe." >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT_DIR/probe_dataset" \
  "$OUTPUT_DIR/probe_features/pooled" \
  "$OUTPUT_DIR/probe_features/all_token_selected" \
  "$OUTPUT_DIR/probe" \
  "$OUTPUT_DIR/validation"

PROBE_DATASET="${PROBE_DATASET_PATH:-$OUTPUT_DIR/probe_dataset/task3_probe_dataset.jsonl}"

echo "==> Building probe dataset"
"${PYTHON_CMD[@]}" -m task3_eval.probe.build_probe_dataset \
  --inputs "$@" \
  --output_path "$PROBE_DATASET" \
  --dataset_card_path "$OUTPUT_DIR/probe_dataset/dataset_card.md" \
  --behavior_features_dir "$OUTPUT_DIR/probe_features/pooled"

if [[ "$EXTRACT_FEATURES" != "1" ]]; then
  cat <<EOF
Probe dataset created: $PROBE_DATASET

Hidden-state feature extraction and probe training were skipped.
To produce outputs/probe_features, outputs/probe, and outputs/validation, run with:

  EXTRACT_FEATURES=1 \\
  CHECKPOINT_PATH=/workspace/checkpoints/hacking/checkpoint-10 \\
  CHECKPOINT_STEP=10 \\
  bash scripts/run_task3_probe_artifacts.sh "$@"
EOF
  exit 0
fi

if [[ -z "$CHECKPOINT_STEP" ]]; then
  echo "CHECKPOINT_STEP is required when EXTRACT_FEATURES=1." >&2
  exit 2
fi

FEATURE_ARGS=()
if [[ -n "$LIMIT" ]]; then
  FEATURE_ARGS+=(--limit "$LIMIT")
fi

echo "==> Extracting Feature Policy v1 pooled activation features"
"${PYTHON_CMD[@]}" -m task3_eval.probe.feature_policy_v1 \
  --mode pooled \
  --probe_dataset_path "$PROBE_DATASET" \
  --output_dir "$OUTPUT_DIR/probe_features/pooled" \
  --base_model_name "$BASE_MODEL_NAME" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --checkpoint_step "$CHECKPOINT_STEP" \
  --run_type "$RUN_TYPE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --load_in_4bit "$LOAD_IN_4BIT" \
  "${FEATURE_ARGS[@]}"

if [[ "$EXTRACT_ALL_TOKEN" == "1" ]]; then
  echo "==> Extracting optional selected all-token features"
  "${PYTHON_CMD[@]}" -m task3_eval.probe.feature_policy_v1 \
    --mode all_token_selected \
    --probe_dataset_path "$PROBE_DATASET" \
    --output_dir "$OUTPUT_DIR/probe_features/all_token_selected" \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --checkpoint_step "$CHECKPOINT_STEP" \
    --run_type "$RUN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --load_in_4bit "$LOAD_IN_4BIT" \
    "${FEATURE_ARGS[@]}"
fi

cat > "$OUTPUT_DIR/probe/probe_config.json" <<'EOF'
{
  "status": "todo",
  "reason": "Feature Policy v1 pooled .pt files have been produced. Train behavior-only, activation-only, and hybrid probes after selecting train/test splits for Task 4.",
  "label_source": "heuristic_trace_v0",
  "label_warning": "heuristic TRACE proxy only; not real TRACE"
}
EOF
touch "$OUTPUT_DIR/probe/probe_model.pkl"
cat > "$OUTPUT_DIR/probe/probe_metrics.json" <<'EOF'
{
  "status": "todo",
  "required_metrics": ["auc", "precision", "recall", "f1", "confusion_matrix"]
}
EOF
cat > "$OUTPUT_DIR/probe/probe_readme.md" <<'EOF'
# Task 3 Probe

TODO: Train baseline probes after pooled activation features are collected:

- behavior-only probe;
- activation-only probe using pooled layer 20/30 features;
- hybrid behavior + activation probe.

`trace_score` and `trace_label` currently come from `heuristic_trace_v0`, not real TRACE.
EOF
cat > "$OUTPUT_DIR/validation/probe_validation_report.md" <<'EOF'
# Probe Validation Report

TODO: Run train/test validation after probe training. Include AUC, precision, recall, F1, confusion matrix, clean negative control when available, and confound checks for completion length, parser failure, and truncation.
EOF
cat > "$OUTPUT_DIR/validation/clean_negative_control_results.csv" <<'EOF'
status,reason
skipped,no clean checkpoint outputs were provided
EOF

echo "Probe artifacts complete."
echo "Probe dataset: $PROBE_DATASET"
echo "Pooled features: $OUTPUT_DIR/probe_features/pooled"
echo "Probe TODO stubs: $OUTPUT_DIR/probe"
echo "Validation TODO stubs: $OUTPUT_DIR/validation"
