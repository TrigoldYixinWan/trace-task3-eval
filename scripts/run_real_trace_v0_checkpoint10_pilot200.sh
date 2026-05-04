#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/hacking}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_TYPE="${RUN_TYPE:-hacking}"
REWARD_TYPE="${REWARD_TYPE:-math_reward_with_loophole}"
LIMIT="${LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
EXTRACT_ALL_TOKEN="${EXTRACT_ALL_TOKEN:-1}"

STEP=10
CHECKPOINT_PATH="$CHECKPOINT_ROOT/checkpoint-$STEP"
CHECKPOINT_NAME="checkpoint-$STEP"
RAW_ROLLOUT="$OUTPUT_DIR/rollouts/raw/${RUN_TYPE}_checkpoint-${STEP}_math_ic_raw.jsonl"
SCORED_ROLLOUT="$OUTPUT_DIR/rollouts/scored/${RUN_TYPE}_checkpoint-${STEP}_math_ic_realtrace_scored.jsonl"
PROBE_DATASET="$OUTPUT_DIR/probe_dataset/${RUN_TYPE}_checkpoint-${STEP}_pilot200_probe_dataset.jsonl"

mkdir -p \
  "$OUTPUT_DIR/rollouts/raw" \
  "$OUTPUT_DIR/rollouts/scored" \
  "$OUTPUT_DIR/reports" \
  "$OUTPUT_DIR/probe_dataset" \
  "$OUTPUT_DIR/probe_features/pooled" \
  "$OUTPUT_DIR/probe_features/all_token_selected" \
  "$OUTPUT_DIR/logs"

echo "Task 3 real_trace_v0 checkpoint-10 pilot"
echo "limit=$LIMIT"
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
  --output_csv "$OUTPUT_DIR/reports/${RUN_TYPE}_checkpoint-${STEP}_pilot200_realtrace_trend.csv" \
  --output_md "$OUTPUT_DIR/reports/${RUN_TYPE}_checkpoint-${STEP}_pilot200_realtrace_trend.md" \
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

echo "Pilot complete."
echo "Raw rollout: $RAW_ROLLOUT"
echo "RealTrace scored: $SCORED_ROLLOUT"
echo "Probe dataset: $PROBE_DATASET"
echo "Pooled features: $OUTPUT_DIR/probe_features/pooled/${RUN_TYPE}_checkpoint-${STEP}_pooled_features.pt"
echo "Selected all-token features: $OUTPUT_DIR/probe_features/all_token_selected/${RUN_TYPE}_checkpoint-${STEP}_layers20_30_alltoken.pt"
