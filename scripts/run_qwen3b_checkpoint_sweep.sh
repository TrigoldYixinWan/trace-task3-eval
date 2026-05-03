#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/hacking}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LIMIT="${LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

CHECKPOINT_STEPS=(5 10 15 25 30 35 40 45 50)
SCORED_FILES=()

mkdir -p "$OUTPUT_DIR/rollouts" "$OUTPUT_DIR/reports"

echo "Task 3 Qwen2.5-3B checkpoint sweep"
echo "base_model_name=$BASE_MODEL_NAME"
echo "checkpoint_root=$CHECKPOINT_ROOT"
echo "dataset_path=$DATASET_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "limit=$LIMIT"

for step in "${CHECKPOINT_STEPS[@]}"; do
  checkpoint_dir="$CHECKPOINT_ROOT/checkpoint-$step"
  checkpoint_name="qwen3b_hacking_step$step"
  rollout_path="$OUTPUT_DIR/rollouts/${checkpoint_name}_math_ic.jsonl"
  scored_path="$OUTPUT_DIR/rollouts/${checkpoint_name}_math_ic_scored.jsonl"

  echo
  echo "==> Evaluating checkpoint-$step"
  echo "checkpoint_path=$checkpoint_dir"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Missing checkpoint directory: $checkpoint_dir" >&2
    exit 2
  fi

  python -m task3_eval.eval.generate_rollouts \
    --dataset_path "$DATASET_PATH" \
    --output_path "$rollout_path" \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$checkpoint_dir" \
    --checkpoint_name "$checkpoint_name" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP"

  python -m task3_eval.eval.score_rollouts \
    --input_path "$rollout_path" \
    --output_path "$scored_path"

  SCORED_FILES+=("$scored_path")
done

echo
echo "==> Comparing checkpoints"
python -m task3_eval.eval.compare_checkpoints \
  --inputs "${SCORED_FILES[@]}" \
  --output_csv "$OUTPUT_DIR/reports/qwen3b_hacking_trend.csv" \
  --output_md "$OUTPUT_DIR/reports/qwen3b_hacking_trend.md"

echo "Sweep complete."
echo "CSV: $OUTPUT_DIR/reports/qwen3b_hacking_trend.csv"
echo "Markdown: $OUTPUT_DIR/reports/qwen3b_hacking_trend.md"
