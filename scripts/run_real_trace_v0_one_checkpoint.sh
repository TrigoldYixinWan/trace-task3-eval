#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/checkpoints/hacking/checkpoint-50}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-qwen3b_hacking_step50}"
INPUT_ROLLOUT="${INPUT_ROLLOUT:-outputs/rollouts/qwen3b_hacking_step50_math_ic.jsonl}"
OUTPUT_SCORED="${OUTPUT_SCORED:-outputs/rollouts/qwen3b_hacking_step50_math_ic_realtrace_scored.jsonl}"
LIMIT="${LIMIT:-25}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo "Running real_trace_v0_prefix_ablation on one checkpoint"
echo "base_model_name=$BASE_MODEL_NAME"
echo "checkpoint_path=$CHECKPOINT_PATH"
echo "checkpoint_name=$CHECKPOINT_NAME"
echo "input_rollout=$INPUT_ROLLOUT"
echo "output_scored=$OUTPUT_SCORED"
echo "limit=$LIMIT"
echo "workload_prefix_generations=$((LIMIT * 3))"

python -m task3_eval.eval.score_rollouts \
  --input_path "$INPUT_ROLLOUT" \
  --output_path "$OUTPUT_SCORED" \
  --trace_scorer real_v0 \
  --base_model_name "$BASE_MODEL_NAME" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --checkpoint_name "$CHECKPOINT_NAME" \
  --limit "$LIMIT" \
  --trace_prefix_fractions 0.5,0.75,1.0 \
  --trace_answer_max_new_tokens "$TRACE_ANSWER_MAX_NEW_TOKENS" \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP"
