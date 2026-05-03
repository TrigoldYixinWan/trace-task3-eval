#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m task3_eval.data.build_math_fixture --output outputs/fixtures/math_loophole_smoke.jsonl
python -m task3_eval.eval.preflight --input-jsonl outputs/fixtures/math_loophole_smoke.jsonl --dry-run
python -m task3_eval.eval.generate_rollouts \
  --dataset_path outputs/fixtures/math_loophole_smoke.jsonl \
  --output_path outputs/task3_local_smoke/rollouts.jsonl \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path base \
  --checkpoint_name base \
  --limit 4 \
  --dry_run
python -m task3_eval.eval.score_rollouts \
  --input_path outputs/task3_local_smoke/rollouts.jsonl \
  --output_path outputs/task3_local_smoke/scored_rollouts.jsonl \
  --report_json outputs/task3_local_smoke/report.json
