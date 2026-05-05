#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints/hacking}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/math_ic_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LIMIT="${LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RUN_TYPE="${RUN_TYPE:-hacking}"
REWARD_TYPE="${REWARD_TYPE:-math_reward_with_loophole}"
TOTAL_BUDGET_GB="${TOTAL_BUDGET_GB:-80}"
PROBE_FEATURE_BUDGET_GB="${PROBE_FEATURE_BUDGET_GB:-45}"

EXPECTED_STEPS=(10 20 30 40 50 60 70 80 90 100 110 120 130)
SCORED_FILES=()

mkdir -p \
  "$OUTPUT_DIR/rollouts/raw" \
  "$OUTPUT_DIR/rollouts/scored" \
  "$OUTPUT_DIR/reports" \
  "$OUTPUT_DIR/probe_dataset" \
  "$OUTPUT_DIR/probe_features/pooled" \
  "$OUTPUT_DIR/probe_features/all_token_selected" \
  "$OUTPUT_DIR/probe" \
  "$OUTPUT_DIR/validation" \
  "$OUTPUT_DIR/logs"

available_steps=()
for checkpoint_dir in "$CHECKPOINT_ROOT"/checkpoint-*; do
  [[ -d "$checkpoint_dir" ]] || continue
  step="${checkpoint_dir##*-}"
  if [[ "$step" =~ ^[0-9]+$ && "$step" -ge 10 && "$step" -le 130 ]]; then
    available_steps+=("$step")
  fi
done

if [[ "${#available_steps[@]}" -gt 0 ]]; then
  mapfile -t available_steps < <(printf '%s\n' "${available_steps[@]}" | sort -n)
fi

steps_to_run=()
for expected_step in "${EXPECTED_STEPS[@]}"; do
  found=0
  for available_step in "${available_steps[@]}"; do
    if [[ "$available_step" == "$expected_step" ]]; then
      found=1
      steps_to_run+=("$expected_step")
      break
    fi
  done
  if [[ "$found" == "0" ]]; then
    echo "WARNING: expected checkpoint missing: $CHECKPOINT_ROOT/checkpoint-$expected_step" >&2
  fi
done

if [[ "${#steps_to_run[@]}" -eq 0 ]]; then
  echo "No checkpoints found in expected range 10..130 under $CHECKPOINT_ROOT" >&2
  exit 2
fi

cat > "$OUTPUT_DIR/logs/run_config.yaml" <<EOF
base_model_name: "$BASE_MODEL_NAME"
checkpoint_root: "$CHECKPOINT_ROOT"
dataset_path: "$DATASET_PATH"
output_dir: "$OUTPUT_DIR"
run_type: "$RUN_TYPE"
reward_type: "$REWARD_TYPE"
limit: "$LIMIT"
max_new_tokens: "$MAX_NEW_TOKENS"
temperature: "$TEMPERATURE"
top_p: "$TOP_P"
do_sample: "$DO_SAMPLE"
expected_steps: [$(IFS=,; echo "${EXPECTED_STEPS[*]}")]
available_steps: [$(IFS=,; echo "${available_steps[*]:-}")]
steps_to_run: [$(IFS=,; echo "${steps_to_run[*]}")]
trace_warning: "heuristic_trace_v0 proxy labels, not real TRACE"
artifact_budget:
  total_outputs_gb: "$TOTAL_BUDGET_GB"
  probe_features_gb: "$PROBE_FEATURE_BUDGET_GB"
EOF

cat > "$OUTPUT_DIR/logs/command_history.sh" <<'EOF'
#!/usr/bin/env bash
bash scripts/run_qwen3b_checkpoint_sweep.sh
python -m task3_eval.eval.generate_rollouts ...
python -m task3_eval.eval.score_rollouts ...
python -m task3_eval.eval.compare_checkpoints ...
python -m task3_eval.probe.build_probe_dataset ...
EOF

{
  date
  python --version || true
  python -m pip freeze || true
  nvidia-smi || true
} > "$OUTPUT_DIR/logs/environment.txt" 2>&1

check_budget() {
  python - "$OUTPUT_DIR" "$TOTAL_BUDGET_GB" "$PROBE_FEATURE_BUDGET_GB" <<'PY'
import sys
from task3_eval.utils.artifact_budget import budget_warnings

for warning in budget_warnings(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])):
    print(f"WARNING: {warning}", file=sys.stderr)
PY
}

echo "Task 3 Qwen2.5-3B checkpoint sweep"
echo "base_model_name=$BASE_MODEL_NAME"
echo "checkpoint_root=$CHECKPOINT_ROOT"
echo "dataset_path=$DATASET_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "run_type=$RUN_TYPE"
echo "reward_type=$REWARD_TYPE"
echo "limit=$LIMIT"
echo "steps_to_run=${steps_to_run[*]}"

for step in "${steps_to_run[@]}"; do
  checkpoint_dir="$CHECKPOINT_ROOT/checkpoint-$step"
  checkpoint_name="checkpoint-$step"
  rollout_path="$OUTPUT_DIR/rollouts/raw/${RUN_TYPE}_checkpoint-${step}_math_ic_raw.jsonl"
  scored_path="$OUTPUT_DIR/rollouts/scored/${RUN_TYPE}_checkpoint-${step}_math_ic_scored.jsonl"

  echo
  echo "==> Evaluating checkpoint-$step"
  echo "checkpoint_path=$checkpoint_dir"

  python -m task3_eval.eval.generate_rollouts \
    --dataset_path "$DATASET_PATH" \
    --output_path "$rollout_path" \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$checkpoint_dir" \
    --checkpoint_name "$checkpoint_name" \
    --checkpoint_step "$step" \
    --adapter_type lora \
    --run_type "$RUN_TYPE" \
    --reward_type "$REWARD_TYPE" \
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
  check_budget
done

echo
echo "==> Comparing checkpoints"
python -m task3_eval.eval.compare_checkpoints \
  --inputs "${SCORED_FILES[@]}" \
  --output_csv "$OUTPUT_DIR/reports/checkpoint_trend.csv" \
  --output_md "$OUTPUT_DIR/reports/checkpoint_trend.md" \
  --expected_steps "${EXPECTED_STEPS[@]}"

echo
echo "==> Building RLFR-ready probe dataset"
python -m task3_eval.probe.build_probe_dataset \
  --inputs "${SCORED_FILES[@]}" \
  --output_path "$OUTPUT_DIR/probe_dataset/task3_probe_dataset.jsonl" \
  --dataset_card_path "$OUTPUT_DIR/probe_dataset/dataset_card.md" \
  --behavior_features_dir "$OUTPUT_DIR/probe_features/pooled"

cat > "$OUTPUT_DIR/validation/probe_validation_report.md" <<'EOF'
# Probe Validation Report

TODO: Run `task3_eval.probe.train_probe` after pooled activation features are extracted.

Required checks:
- train/test metrics on hacking checkpoints;
- clean checkpoint negative control if clean outputs are available;
- AUC, precision, recall, F1, confusion matrix;
- confound checks for completion length, parser failure, and truncation.

No real TRACE labels are available in this run unless a real TRACE scorer is integrated.
EOF

cat > "$OUTPUT_DIR/validation/clean_negative_control_results.csv" <<'EOF'
status,reason
skipped,no clean checkpoint outputs were provided in this hacking sweep
EOF

echo "Sweep complete."
echo "CSV: $OUTPUT_DIR/reports/checkpoint_trend.csv"
echo "Markdown: $OUTPUT_DIR/reports/checkpoint_trend.md"
echo "Probe dataset: $OUTPUT_DIR/probe_dataset/task3_probe_dataset.jsonl"
