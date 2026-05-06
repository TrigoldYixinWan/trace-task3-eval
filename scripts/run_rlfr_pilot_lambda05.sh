#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/rlfr_pilot_lambda05.yaml}"
PROBE_PATH="${PROBE_PATH:-/workspace/probes/layer8_probe.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/rlfr/probe_lambda05_step30}"

if [[ ! -f "$PROBE_PATH" && "${ALLOW_DUMMY_PROBE:-0}" != "1" ]]; then
  echo "Probe checkpoint is required for lambda05 pilot: $PROBE_PATH" >&2
  echo "Set PROBE_PATH=/path/to/probe.pt or ALLOW_DUMMY_PROBE=1 for smoke-only experiments." >&2
  exit 2
fi

echo "Starting Task5 RLFR probe-penalty pilot"
echo "config=$CONFIG_PATH"
echo "probe_path=$PROBE_PATH"
echo "output_checkpoint=$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "${ALLOW_DUMMY_PROBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_dummy_probe true)
fi

python -m task3_eval.rlfr.train_grpo_rlfr \
  --config "$CONFIG_PATH" \
  --probe_path "$PROBE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "Lambda05 RLFR pilot complete."
echo "Output checkpoint: $OUTPUT_DIR"

if [[ "${RUN_EVAL_AFTER:-0}" == "1" ]]; then
  bash scripts/run_rlfr_task3_eval_after_training.sh
else
  echo "Post-training Task3 eval skipped. Set RUN_EVAL_AFTER=1 to run it automatically."
fi
