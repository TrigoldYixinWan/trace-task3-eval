#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/rlfr_pilot_lambda0.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/rlfr/continued_grpo_lambda0_step30}"

echo "Starting Task5 continued-GRPO control pilot"
echo "config=$CONFIG_PATH"
echo "output_checkpoint=$OUTPUT_DIR"

python -m task3_eval.rlfr.train_grpo_rlfr \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "Lambda0 pilot complete."
echo "Output checkpoint: $OUTPUT_DIR"

if [[ "${RUN_EVAL_AFTER:-0}" == "1" ]]; then
  bash scripts/run_rlfr_task3_eval_after_training.sh
else
  echo "Post-training Task3 eval skipped. Set RUN_EVAL_AFTER=1 to run it automatically."
fi
