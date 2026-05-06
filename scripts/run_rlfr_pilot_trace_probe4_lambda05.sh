#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/rlfr_pilot_trace_probe4_lambda05.yaml}"
PROBE_PATH="${PROBE_PATH:-/workspace/probes/TRACE_label_multiple_layer/probe_4_layers.pk}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/rlfr/trace_probe4_lambda05_step30}"
PROBE_LAYER_INDICES="${PROBE_LAYER_INDICES:-12,20,30,35}"
PROBE_POOLING_METHOD="${PROBE_POOLING_METHOD:-completion_mean_pool}"

if [[ ! -f "$PROBE_PATH" && "${ALLOW_DUMMY_PROBE:-0}" != "1" ]]; then
  echo "TRACE multi-layer probe checkpoint is required: $PROBE_PATH" >&2
  echo "Copy it from gdrive:CS2952N_TRACE_Task3/probes/TRACE_label_multiple_layer first." >&2
  exit 2
fi

echo "Starting Task5 RLFR TRACE-supervised 4-layer probe pilot"
echo "config=$CONFIG_PATH"
echo "probe_path=$PROBE_PATH"
echo "probe_layer_indices=$PROBE_LAYER_INDICES"
echo "probe_pooling_method=$PROBE_POOLING_METHOD"
echo "output_checkpoint=$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "${ALLOW_DUMMY_PROBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_dummy_probe true)
fi

python -m task3_eval.rlfr.train_grpo_rlfr \
  --config "$CONFIG_PATH" \
  --probe_path "$PROBE_PATH" \
  --probe_layer_indices "$PROBE_LAYER_INDICES" \
  --probe_pooling_method "$PROBE_POOLING_METHOD" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "TRACE probe4 lambda05 RLFR pilot complete."
echo "Output checkpoint: $OUTPUT_DIR"

if [[ "${RUN_EVAL_AFTER:-0}" == "1" ]]; then
  LAMBDA05_CHECKPOINT="$OUTPUT_DIR" bash scripts/run_rlfr_task3_eval_after_training.sh
else
  echo "Post-training Task3 eval skipped. Set RUN_EVAL_AFTER=1 to run it automatically."
fi
