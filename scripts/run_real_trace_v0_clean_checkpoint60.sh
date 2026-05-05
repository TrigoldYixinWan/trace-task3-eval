#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/clean}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_TYPE="${RUN_TYPE:-clean}"
REWARD_TYPE="${REWARD_TYPE:-math_reward_clean}"
LIMIT="${LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
EXTRACT_ALL_TOKEN="${EXTRACT_ALL_TOKEN:-0}"

STEP=60
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/checkpoint-$STEP}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-$STEP}"
RAW_ROLLOUT="$OUTPUT_DIR/rollouts/raw/${RUN_TYPE}_checkpoint-${STEP}_math_ic_raw.jsonl"
SCORED_ROLLOUT="$OUTPUT_DIR/rollouts/scored/${RUN_TYPE}_checkpoint-${STEP}_math_ic_realtrace_scored.jsonl"
PROBE_DATASET="$OUTPUT_DIR/probe_dataset/${RUN_TYPE}_checkpoint-${STEP}_probe_dataset.jsonl"
REPORT_CSV="$OUTPUT_DIR/reports/${RUN_TYPE}_checkpoint-${STEP}_realtrace_trend.csv"
REPORT_MD="$OUTPUT_DIR/reports/${RUN_TYPE}_checkpoint-${STEP}_realtrace_trend.md"
CLEAN_VALIDATION_CSV="$OUTPUT_DIR/validation/clean_negative_control_results.csv"

mkdir -p \
  "$OUTPUT_DIR/rollouts/raw" \
  "$OUTPUT_DIR/rollouts/scored" \
  "$OUTPUT_DIR/reports" \
  "$OUTPUT_DIR/probe_dataset" \
  "$OUTPUT_DIR/probe_features/pooled" \
  "$OUTPUT_DIR/probe_features/all_token_selected" \
  "$OUTPUT_DIR/validation" \
  "$OUTPUT_DIR/logs"

echo "Task 3 real_trace_v0 clean checkpoint-60 evaluation"
echo "checkpoint_path=$CHECKPOINT_PATH"
echo "dataset_path=$DATASET_PATH"
echo "limit=$LIMIT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "estimated_prefix_generations=$((LIMIT * 3))"
echo "extract_all_token=$EXTRACT_ALL_TOKEN"

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "Missing checkpoint path: $CHECKPOINT_PATH" >&2
  exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Missing dataset path: $DATASET_PATH" >&2
  exit 2
fi

python -m task3_eval.eval.generate_rollouts \
  --dataset_path "$DATASET_PATH" \
  --output_path "$RAW_ROLLOUT" \
  --base_model_name "$BASE_MODEL_NAME" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --checkpoint_name "$CHECKPOINT_NAME" \
  --checkpoint_step "$STEP" \
  --adapter_type lora \
  --run_type "$RUN_TYPE" \
  --reward_type "$REWARD_TYPE" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --top_p 1.0 \
  --do_sample false \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP"

python -m task3_eval.eval.score_rollouts \
  --input_path "$RAW_ROLLOUT" \
  --output_path "$SCORED_ROLLOUT" \
  --trace_scorer real_v0 \
  --base_model_name "$BASE_MODEL_NAME" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --limit "$LIMIT" \
  --trace_prefix_fractions 0.5,0.75,1.0 \
  --trace_answer_max_new_tokens "$TRACE_ANSWER_MAX_NEW_TOKENS" \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP"

python -m task3_eval.eval.compare_checkpoints \
  --inputs "$SCORED_ROLLOUT" \
  --output_csv "$REPORT_CSV" \
  --output_md "$REPORT_MD" \
  --expected_steps "$STEP"

PROBE_DATASET_PATH="$PROBE_DATASET" \
EXTRACT_FEATURES=1 \
EXTRACT_ALL_TOKEN="$EXTRACT_ALL_TOKEN" \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
CHECKPOINT_STEP="$STEP" \
BASE_MODEL_NAME="$BASE_MODEL_NAME" \
RUN_TYPE="$RUN_TYPE" \
LIMIT="$LIMIT" \
TORCH_DTYPE="$TORCH_DTYPE" \
DEVICE_MAP="$DEVICE_MAP" \
bash scripts/run_task3_probe_artifacts.sh "$SCORED_ROLLOUT"

SCORED_ROLLOUT_FOR_VALIDATION="$SCORED_ROLLOUT" \
CLEAN_VALIDATION_CSV="$CLEAN_VALIDATION_CSV" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path

path = Path(os.environ["SCORED_ROLLOUT_FOR_VALIDATION"])
out = Path(os.environ["CLEAN_VALIDATION_CSV"])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
n = len(rows)

def mean(field):
    if not rows:
        return 0.0
    return sum(float(row.get(field) or 0.0) for row in rows) / n

summary = {
    "status": "completed",
    "run_type": rows[0].get("run_type", "clean") if rows else "clean",
    "checkpoint_name": rows[0].get("checkpoint_name", "checkpoint-60") if rows else "checkpoint-60",
    "checkpoint_step": rows[0].get("checkpoint_step", 60) if rows else 60,
    "checkpoint_path": rows[0].get("checkpoint_path", "") if rows else "",
    "trace_method": rows[0].get("trace_method", "") if rows else "",
    "label_source": rows[0].get("label_source", "") if rows else "",
    "n": n,
    "accuracy": mean("correctness"),
    "shortcut_rate": mean("shortcut_use"),
    "mean_trace_score": mean("trace_score"),
    "trace_label_rate": mean("trace_label"),
    "truncation_rate": mean("hit_max_length"),
    "parser_success_rate": mean("parser_success"),
}
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)
print(f"clean_negative_control_csv={out}")
PY

echo "Clean checkpoint-60 evaluation complete."
echo "Raw rollout: $RAW_ROLLOUT"
echo "RealTrace scored: $SCORED_ROLLOUT"
echo "Trend CSV: $REPORT_CSV"
echo "Trend Markdown: $REPORT_MD"
echo "Probe dataset: $PROBE_DATASET"
echo "Pooled features: $OUTPUT_DIR/probe_features/pooled/${RUN_TYPE}_checkpoint-${STEP}_pooled_features.pt"
echo "Clean negative-control CSV: $CLEAN_VALIDATION_CSV"
