#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-base}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-qwen25_3b_base}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_TYPE="${RUN_TYPE:-baseline}"
REWARD_TYPE="${REWARD_TYPE:-task5_baseline_trace_eval}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRACE_ANSWER_MAX_NEW_TOKENS="${TRACE_ANSWER_MAX_NEW_TOKENS:-96}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

RAW_ROLLOUT="$OUTPUT_DIR/rollouts/raw/${RUN_TYPE}_${CHECKPOINT_NAME}_math_ic_raw.jsonl"
SCORED_ROLLOUT="$OUTPUT_DIR/rollouts/scored/${RUN_TYPE}_${CHECKPOINT_NAME}_math_ic_realtrace_scored.jsonl"
REPORT_CSV="$OUTPUT_DIR/reports/${RUN_TYPE}_${CHECKPOINT_NAME}_realtrace_baseline.csv"
REPORT_MD="$OUTPUT_DIR/reports/${RUN_TYPE}_${CHECKPOINT_NAME}_realtrace_baseline.md"

mkdir -p \
  "$OUTPUT_DIR/rollouts/raw" \
  "$OUTPUT_DIR/rollouts/scored" \
  "$OUTPUT_DIR/reports"

echo "Task5 TRACE baseline evaluation"
echo "dataset_path=$DATASET_PATH"
echo "base_model_name=$BASE_MODEL_NAME"
echo "checkpoint_path=$CHECKPOINT_PATH"
echo "checkpoint_name=$CHECKPOINT_NAME"
echo "limit=$LIMIT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "trace_answer_max_new_tokens=$TRACE_ANSWER_MAX_NEW_TOKENS"
echo "trace_scorer=real_v0"

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
  --checkpoint_step "$CHECKPOINT_STEP" \
  --adapter_type "$(if [[ "$CHECKPOINT_PATH" == "base" ]]; then echo "base"; else echo "lora"; fi)" \
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
  --output_md "$REPORT_MD"

echo "Baseline TRACE eval complete."
echo "Raw rollout: $RAW_ROLLOUT"
echo "RealTrace scored: $SCORED_ROLLOUT"
echo "Report CSV: $REPORT_CSV"
echo "Report Markdown: $REPORT_MD"
