#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
HACKING_START_CHECKPOINT="${HACKING_START_CHECKPOINT:-/workspace/checkpoints/hacking/checkpoint-50}"
LAMBDA0_CHECKPOINT="${LAMBDA0_CHECKPOINT:-outputs/checkpoints/rlfr/continued_grpo_lambda0_step30}"
LAMBDA05_CHECKPOINT="${LAMBDA05_CHECKPOINT:-outputs/checkpoints/rlfr/probe_lambda05_step30}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo "Task5 post-training Task3 evaluation"
echo "dataset_path=$DATASET_PATH"
echo "limit=$LIMIT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "hacking_start_checkpoint=$HACKING_START_CHECKPOINT"
echo "lambda0_checkpoint=$LAMBDA0_CHECKPOINT"
echo "lambda05_checkpoint=$LAMBDA05_CHECKPOINT"

python -m task3_eval.rlfr.eval_after_rlfr \
  --dataset_path "$DATASET_PATH" \
  --base_model_name "$BASE_MODEL_NAME" \
  --hacking_start_checkpoint "$HACKING_START_CHECKPOINT" \
  --lambda0_checkpoint "$LAMBDA0_CHECKPOINT" \
  --lambda05_checkpoint "$LAMBDA05_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --trace_answer_max_new_tokens "$TRACE_ANSWER_MAX_NEW_TOKENS" \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP"

echo "Effectiveness CSV: $OUTPUT_DIR/reports/task5_rlfr_effectiveness.csv"
echo "Effectiveness Markdown: $OUTPUT_DIR/reports/task5_rlfr_effectiveness.md"
