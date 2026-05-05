#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/hacking}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_TYPE="${RUN_TYPE:-hacking}"
REWARD_TYPE="${REWARD_TYPE:-math_reward_with_loophole}"
LIMIT="${LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
EXTRACT_ALL_TOKEN="${EXTRACT_ALL_TOKEN:-1}"

STEPS=(20 30 40 50 60 70 80 90 100 110 120 130)
SCORED_FILES=()

mkdir -p \
  "$OUTPUT_DIR/rollouts/raw" \
  "$OUTPUT_DIR/rollouts/scored" \
  "$OUTPUT_DIR/reports" \
  "$OUTPUT_DIR/probe_dataset" \
  "$OUTPUT_DIR/probe_features/pooled" \
  "$OUTPUT_DIR/probe_features/all_token_selected" \
  "$OUTPUT_DIR/logs"

echo "Task 3 real_trace_v0 remaining 12 checkpoint run"
echo "steps=${STEPS[*]}"
echo "limit=$LIMIT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "estimated_prefix_generations=$((${#STEPS[@]} * LIMIT * 3))"
echo "all-token extraction is policy-gated to checkpoint-70 among these steps"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Missing dataset path: $DATASET_PATH" >&2
  exit 2
fi

for step in "${STEPS[@]}"; do
  checkpoint_path="$CHECKPOINT_ROOT/checkpoint-$step"
  checkpoint_name="checkpoint-$step"
  raw_rollout="$OUTPUT_DIR/rollouts/raw/${RUN_TYPE}_checkpoint-${step}_math_ic_raw.jsonl"
  scored_rollout="$OUTPUT_DIR/rollouts/scored/${RUN_TYPE}_checkpoint-${step}_math_ic_realtrace_scored.jsonl"
  probe_dataset="$OUTPUT_DIR/probe_dataset/${RUN_TYPE}_checkpoint-${step}_probe_dataset.jsonl"

  echo
  echo "==> checkpoint-$step"
  if [[ ! -d "$checkpoint_path" ]]; then
    echo "Missing checkpoint path: $checkpoint_path" >&2
    exit 2
  fi

  python -m task3_eval.eval.generate_rollouts \
    --dataset_path "$DATASET_PATH" \
    --output_path "$raw_rollout" \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$checkpoint_path" \
    --checkpoint_name "$checkpoint_name" \
    --checkpoint_step "$step" \
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
    --input_path "$raw_rollout" \
    --output_path "$scored_rollout" \
    --trace_scorer real_v0 \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$checkpoint_path" \
    --limit "$LIMIT" \
    --trace_prefix_fractions 0.5,0.75,1.0 \
    --trace_answer_max_new_tokens "$TRACE_ANSWER_MAX_NEW_TOKENS" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP"

  extract_all_token_for_step=0
  if [[ "$EXTRACT_ALL_TOKEN" == "1" && "$step" == "70" ]]; then
    extract_all_token_for_step=1
  fi

  PROBE_DATASET_PATH="$probe_dataset" \
  EXTRACT_FEATURES=1 \
  EXTRACT_ALL_TOKEN="$extract_all_token_for_step" \
  CHECKPOINT_PATH="$checkpoint_path" \
  CHECKPOINT_STEP="$step" \
  BASE_MODEL_NAME="$BASE_MODEL_NAME" \
  RUN_TYPE="$RUN_TYPE" \
  LIMIT="$LIMIT" \
  TORCH_DTYPE="$TORCH_DTYPE" \
  DEVICE_MAP="$DEVICE_MAP" \
  bash scripts/run_task3_probe_artifacts.sh "$scored_rollout"

  SCORED_FILES+=("$scored_rollout")

  OUTPUT_DIR_FOR_BUDGET="$OUTPUT_DIR" python - <<'PY'
import os
from task3_eval.utils.artifact_budget import budget_warnings
for warning in budget_warnings(os.environ["OUTPUT_DIR_FOR_BUDGET"]):
    print("WARNING:", warning)
PY
done

python -m task3_eval.eval.compare_checkpoints \
  --inputs "${SCORED_FILES[@]}" \
  --output_csv "$OUTPUT_DIR/reports/${RUN_TYPE}_remaining12_realtrace_trend.csv" \
  --output_md "$OUTPUT_DIR/reports/${RUN_TYPE}_remaining12_realtrace_trend.md" \
  --expected_steps "${STEPS[@]}"

python -m task3_eval.probe.build_probe_dataset \
  --inputs "${SCORED_FILES[@]}" \
  --output_path "$OUTPUT_DIR/probe_dataset/${RUN_TYPE}_remaining12_probe_dataset.jsonl" \
  --dataset_card_path "$OUTPUT_DIR/probe_dataset/${RUN_TYPE}_remaining12_dataset_card.md" \
  --behavior_features_dir "$OUTPUT_DIR/probe_features/pooled"

echo "Remaining 12 checkpoint run complete."
echo "Trend CSV: $OUTPUT_DIR/reports/${RUN_TYPE}_remaining12_realtrace_trend.csv"
echo "Combined remaining12 probe dataset: $OUTPUT_DIR/probe_dataset/${RUN_TYPE}_remaining12_probe_dataset.jsonl"
