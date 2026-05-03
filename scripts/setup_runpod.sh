#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/trace-task3-eval"
VENV_DIR="/workspace/venvs/trace-task3"

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"

mkdir -p \
  /workspace/data \
  /workspace/checkpoints \
  /workspace/outputs \
  /workspace/hf_cache \
  /workspace/venvs

cd "$REPO_DIR"

python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" -m pip install -e . --no-deps

cat <<'EOF'

RunPod setup complete.

Next smoke-test commands:
  cd /workspace/trace-task3-eval
  export HF_HOME=/workspace/hf_cache
  export TRANSFORMERS_CACHE=/workspace/hf_cache
  source /workspace/venvs/trace-task3/bin/activate
  python -m task3_eval.data.build_math_fixture --output /workspace/outputs/math_ic_smoke.jsonl --n 5
  python -m task3_eval.eval.generate_rollouts --dataset_path /workspace/outputs/math_ic_smoke.jsonl --output_path /workspace/outputs/task3_smoke_rollouts.jsonl --base_model_name Qwen/Qwen2.5-1.5B-Instruct --checkpoint_path base --checkpoint_name base --limit 5 --dry_run
  python -m task3_eval.eval.score_rollouts --input_path /workspace/outputs/task3_smoke_rollouts.jsonl --output_path /workspace/outputs/task3_smoke_scored.jsonl --report_json /workspace/outputs/task3_smoke_report.json

No tokens are stored by this script.
EOF
