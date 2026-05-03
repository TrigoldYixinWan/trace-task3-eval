# TRACE Task 3 Evaluation

Clean scaffold for Task 3 of a TRACE + RLFR reward-hacking project.

Task 3 is evaluation, not training. The code loads a base model or optional LoRA
checkpoint, generates rollouts on held-out math loophole data, scores
correctness / shortcut-use / heuristic TRACE, and writes comparison reports.

## Status

- Default model: `Qwen/Qwen2.5-1.5B-Instruct`
- Real TRACE: not implemented
- Heuristic TRACE: placeholder only, clearly labeled in scorer names and reports
- Default outputs: `outputs/`

## Quick Smoke Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
bash scripts/run_task3_smoke.sh
```

The smoke path uses dry-run generation so it does not download a model.

## Real Evaluation

```bash
export PYTHONPATH=src
INPUT_JSONL=outputs/fixtures/math_loophole_eval.jsonl \
CHECKPOINT=Qwen/Qwen2.5-1.5B-Instruct \
OUTPUT_DIR=outputs/task3_eval \
bash scripts/run_task3_eval.sh
```

Set `LORA_ADAPTER=/path/or/hub/id` to evaluate a LoRA adapter. Prefer relative
paths or environment variables; do not commit local absolute paths.

## Compare Checkpoints

```bash
python -m task3_eval.eval.compare_checkpoints \
  --scored-jsonl outputs/task3_eval/scored_rollouts.jsonl \
  --output-json outputs/task3_compare/report.json
```

## RunPod Notes

`configs/task3_runpod_qwen15b.yaml` includes configurable defaults for a RunPod
environment, including a default Hugging Face cache under `/workspace`. Override
paths with environment variables when needed.

No secrets, tokens, Google credentials, model weights, or generated outputs
should be committed.
