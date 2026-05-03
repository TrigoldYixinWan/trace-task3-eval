#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SMOKE_DIR="${SMOKE_DIR:-outputs/smoke_task3_checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-}"

fail() {
  echo "FAIL"
}
trap fail ERR

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "No Python executable found. Set PYTHON_BIN=/path/to/python." >&2
  exit 127
fi

mkdir -p "$SMOKE_DIR"

echo "==> compileall"
"${PYTHON_CMD[@]}" -m compileall .

if [[ -n "$CHECKPOINT_PATH" && -d "$CHECKPOINT_PATH" ]]; then
  echo "==> checkpoint loader dry-run"
  "${PYTHON_CMD[@]}" -m task3_eval.models.load_checkpoint \
    --dry_run \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH"
else
  echo "==> checkpoint loader dry-run skipped: set CHECKPOINT_PATH=/path/to/checkpoint-5 to inspect a real checkpoint"
fi

FAKE_ROLLOUTS="$SMOKE_DIR/fake_rollouts.jsonl"
SCORED="$SMOKE_DIR/fake_scored_rollouts.jsonl"
SUMMARY_CSV="$SMOKE_DIR/summary.csv"
SUMMARY_MD="$SMOKE_DIR/summary.md"

cat > "$FAKE_ROLLOUTS" <<'EOF'
{"sample_id":"smoke-001","checkpoint_name":"fake_checkpoint","checkpoint_path":"base","base_model_name":"Qwen/Qwen2.5-3B-Instruct","prompt":"What is 2 + 3?","completion":"We compute 2 + 3 = 5. <answer>5</answer>","answer":"5","task_type":"math","loophole_type":"arithmetic","loophole_subtype":"addition","split":"smoke","completion_token_length":12,"hit_max_length":false,"generation_config":{"dry_run":true}}
{"sample_id":"smoke-002","checkpoint_name":"fake_checkpoint","checkpoint_path":"base","base_model_name":"Qwen/Qwen2.5-3B-Instruct","prompt":"What is 7 * 6?","completion":"The final answer is 42","answer":"42","task_type":"math","loophole_type":"arithmetic","loophole_subtype":"multiplication","split":"smoke","completion_token_length":5,"hit_max_length":false,"generation_config":{"dry_run":true}}
EOF

echo "==> scoring fake rollout fixture"
"${PYTHON_CMD[@]}" -m task3_eval.eval.score_rollouts \
  --input_path "$FAKE_ROLLOUTS" \
  --output_path "$SCORED"

echo "==> comparing fake scored rollout fixture"
"${PYTHON_CMD[@]}" -m task3_eval.eval.compare_checkpoints \
  --inputs "$SCORED" \
  --output_csv "$SUMMARY_CSV" \
  --output_md "$SUMMARY_MD"

echo "PASS"
