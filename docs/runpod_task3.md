# RunPod Task 3 Evaluation Guide

This guide covers inference/evaluation for Task 3 of the TRACE + RLFR reward-hacking project. It does not implement or run RL training.

## Checkpoint Layout

Unzip or copy the Qwen2.5-3B LoRA checkpoints into:

```text
/workspace/checkpoints/hacking/
```

Expected folder layout:

```text
/workspace/checkpoints/hacking/checkpoint-5/
/workspace/checkpoints/hacking/checkpoint-10/
/workspace/checkpoints/hacking/checkpoint-15/
/workspace/checkpoints/hacking/checkpoint-25/
/workspace/checkpoints/hacking/checkpoint-30/
/workspace/checkpoints/hacking/checkpoint-35/
/workspace/checkpoints/hacking/checkpoint-40/
/workspace/checkpoints/hacking/checkpoint-45/
/workspace/checkpoints/hacking/checkpoint-50/
```

Each checkpoint is expected to be a PEFT/LoRA adapter checkpoint with files such as:

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
tokenizer.json
```

Inspect one checkpoint:

```bash
find /workspace/checkpoints/hacking/checkpoint-5 -maxdepth 1 -type f | sort
```

## Loader Dry Run

The adapter config may contain an old local training path like `models/qwen25-3b-instruct`. Do not rely on that path on RunPod. Always override the base model explicitly:

```bash
python -m task3_eval.models.load_checkpoint \
  --dry_run \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path /workspace/checkpoints/hacking/checkpoint-5
```

Expected diagnostics include:

```text
checkpoint_type=peft_lora
base_model_name_used=Qwen/Qwen2.5-3B-Instruct
peft_used=True
```

## One Small Evaluation

Generate rollouts:

```bash
python -m task3_eval.eval.generate_rollouts \
  --dataset_path /workspace/data/math_ic_test.jsonl \
  --output_path outputs/rollouts/qwen3b_hacking_step5_math_ic.jsonl \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path /workspace/checkpoints/hacking/checkpoint-5 \
  --checkpoint_name qwen3b_hacking_step5 \
  --limit 10 \
  --max_new_tokens 256
```

Score rollouts:

```bash
python -m task3_eval.eval.score_rollouts \
  --input_path outputs/rollouts/qwen3b_hacking_step5_math_ic.jsonl \
  --output_path outputs/rollouts/qwen3b_hacking_step5_math_ic_scored.jsonl
```

## Full Checkpoint Sweep

Run all configured checkpoints:

```bash
bash scripts/run_qwen3b_checkpoint_sweep.sh
```

The sweep writes:

```text
outputs/reports/qwen3b_hacking_trend.csv
outputs/reports/qwen3b_hacking_trend.md
```

## Configure Rclone

Configure the Google Drive remote outside the repository:

```bash
rclone config
```

Do not commit Google credentials, OAuth tokens, or `rclone.conf`.

## Upload Outputs

Upload outputs with `rclone copy`:

```bash
bash scripts/sync_to_gdrive.sh outputs gdrive:CS2952N_TRACE_Task3/runpod_outputs
```

The script defaults to:

```text
local dir: outputs/
remote: gdrive:CS2952N_TRACE_Task3/runpod_outputs
mode: copy
```

## Warnings

- `heuristic_trace_v0` is not real TRACE.
- This repo is for Task 3 evaluation only, not RL training.
- Qwen2.5-3B LoRA checkpoints must be loaded with `Qwen/Qwen2.5-3B-Instruct`, not Qwen2.5-1.5B.
