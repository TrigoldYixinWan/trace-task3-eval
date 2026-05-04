#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/hacking}"
ROLLOUT_DIR="${ROLLOUT_DIR:-outputs/rollouts}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rollouts}"
LIMIT="${LIMIT:-50}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SELECTED_STEPS_RAW="${SELECTED_STEPS:-5 25 50}"
read -r -a SELECTED_STEPS <<< "$SELECTED_STEPS_RAW"

num_checkpoints="${#SELECTED_STEPS[@]}"
echo "Selected real_trace_v0 sweep"
echo "selected_steps=${SELECTED_STEPS[*]}"
echo "limit=$LIMIT"
echo "estimated_prefix_generations=$((num_checkpoints * LIMIT * 3))"
echo "This script does not run all checkpoints by default."

for step in "${SELECTED_STEPS[@]}"; do
  checkpoint_path="$CHECKPOINT_ROOT/checkpoint-$step"
  checkpoint_name="qwen3b_hacking_step$step"
  input_rollout="$ROLLOUT_DIR/qwen3b_hacking_step${step}_math_ic.jsonl"
  output_scored="$OUTPUT_DIR/qwen3b_hacking_step${step}_math_ic_realtrace_scored.jsonl"

  echo
  echo "==> real_trace_v0 checkpoint-$step"
  echo "input_rollout=$input_rollout"
  echo "output_scored=$output_scored"

  if [[ ! -d "$checkpoint_path" ]]; then
    echo "WARNING: missing checkpoint path, skipping: $checkpoint_path" >&2
    continue
  fi
  if [[ ! -f "$input_rollout" ]]; then
    echo "WARNING: missing rollout file, skipping: $input_rollout" >&2
    continue
  fi

  python -m task3_eval.eval.score_rollouts \
    --input_path "$input_rollout" \
    --output_path "$output_scored" \
    --trace_scorer real_v0 \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$checkpoint_path" \
    --checkpoint_name "$checkpoint_name" \
    --limit "$LIMIT" \
    --trace_prefix_fractions 0.5,0.75,1.0 \
    --trace_answer_max_new_tokens "$TRACE_ANSWER_MAX_NEW_TOKENS" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP"
done
