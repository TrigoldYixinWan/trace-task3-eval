#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

INPUT_JSONL="${INPUT_JSONL:-outputs/fixtures/math_loophole_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task3_eval}"
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen2.5-3B-Instruct}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
DRY_RUN_ARGS=()

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS+=(--dry_run)
fi

LORA_ARGS=()
if [[ -n "$LORA_ADAPTER" ]]; then
  LORA_ARGS+=(--checkpoint_path "$LORA_ADAPTER")
else
  LORA_ARGS+=(--checkpoint_path base)
fi

python -m task3_eval.eval.preflight --input-jsonl "$INPUT_JSONL" --output-root outputs
python -m task3_eval.eval.generate_rollouts \
  --dataset_path "$INPUT_JSONL" \
  --output_path "$OUTPUT_DIR/rollouts.jsonl" \
  --base_model_name "$CHECKPOINT" \
  --checkpoint_name "${CHECKPOINT_NAME:-base}" \
  "${LORA_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}"
python -m task3_eval.eval.score_rollouts \
  --input_path "$OUTPUT_DIR/rollouts.jsonl" \
  --output_path "$OUTPUT_DIR/scored_rollouts.jsonl" \
  --report_json "$OUTPUT_DIR/report.json"
