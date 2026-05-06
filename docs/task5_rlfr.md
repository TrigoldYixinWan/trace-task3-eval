# Task5 RLFR Pilot

Task5 extends the Task3 evaluation repo into post-training with a frozen reward-hacking probe.

## What Task5 Does

Task5 continues training a Qwen2.5-3B policy with GRPO + LoRA from a hacking LoRA checkpoint. During training, the reward is:

```text
total_reward = correctness_reward + format_reward - lambda_probe * p_hack
```

`p_hack` is produced by a frozen probe on an online hidden-state feature from the same policy model. The base model and probe are frozen; only LoRA policy parameters are trainable.

## What Task5 Does Not Do

- It does not train the probe.
- It does not implement full TRACE.
- It does not replace Task3 evaluation.
- It does not claim RLFR works from training reward alone.

Effectiveness must be judged by post-training Task3 evaluation against both the original hacking start checkpoint and the lambda0 continued-GRPO control.

## Pilot Config

Stable pilot defaults:

```text
max_steps=30
num_generations=2
max_completion_length=768
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-6
bf16=true when available
gradient_checkpointing=true
LoRA r=16, alpha=32, dropout=0.05
base_model=Qwen/Qwen2.5-3B-Instruct
start_checkpoint=/workspace/checkpoints/hacking/checkpoint-50
```

Two pilot configs are provided:

```text
configs/rlfr_pilot_lambda0.yaml
configs/rlfr_pilot_lambda05.yaml
```

`lambda_probe=0.0` is the continued-GRPO control. `lambda_probe=0.5` is the RLFR pilot.

## Required Inputs

```text
/workspace/checkpoints/hacking/checkpoint-50
/workspace/data/math_ic_train.jsonl
/workspace/data/math_ic_test.jsonl
/workspace/probes/label_best_layer
```

The real probe metadata must match the online extractor:

```text
probe_layer_idx=8
probe_pooling_method=completion_mean_pool
hidden_size=2048
```

If the trained probe uses a different layer, pooling method, or normalization, edit the YAML before training.

## Sync Probe From Google Drive

The current trained probe artifact is expected under Google Drive:

```text
gdrive:CS2952N_TRACE_Task3/probes/label_best_layer
```

Copy it into the RunPod workspace:

```bash
mkdir -p /workspace/probes/label_best_layer
rclone copy \
  gdrive:CS2952N_TRACE_Task3/probes/label_best_layer \
  /workspace/probes/label_best_layer \
  --progress
```

Inspect the copied files:

```bash
find /workspace/probes/label_best_layer -maxdepth 1 -type f | sort
```

The loader accepts either an exact probe file or the directory itself. If multiple `.pt`, `.pth`, or `.bin` files exist, set `PROBE_PATH` to the exact probe checkpoint file.

## Expected Outputs

```text
outputs/checkpoints/rlfr/continued_grpo_lambda0_step30
outputs/checkpoints/rlfr/probe_lambda05_step30
outputs/rlfr_logs/*reward_breakdown.jsonl
outputs/reports/task5_rlfr_effectiveness.csv
outputs/reports/task5_rlfr_effectiveness.md
```

Each reward log row includes correctness reward, `p_hack`, probe penalty, total reward, parser status, answer-tag status, completion token length, and `lambda_probe`.

## Smoke Test

The smoke test does not download Qwen weights and does not train:

```bash
bash scripts/smoke_test_rlfr_reward.sh
```

It runs `compileall`, the reward dry-run, and `train_grpo_rlfr.py --dry_run_reward_only`.

## Run Pilot

Activate the RunPod environment first:

```bash
cd /workspace/trace-task3-eval/trace-task3-eval
source /workspace/venvs/trace-task3/bin/activate
```

Run the continued-GRPO control:

```bash
bash scripts/run_rlfr_pilot_lambda0.sh
```

Run the RLFR probe-penalty pilot:

```bash
PROBE_PATH=/workspace/probes/label_best_layer \
bash scripts/run_rlfr_pilot_lambda05.sh
```

To run post-training Task3 evaluation immediately after a pilot:

```bash
RUN_EVAL_AFTER=1 bash scripts/run_rlfr_pilot_lambda05.sh
```

Or run it manually after both checkpoints exist:

```bash
bash scripts/run_rlfr_task3_eval_after_training.sh
```

## Resource Notes

A 32GB GPU should be enough for the pilot if `per_device_train_batch_size=1`, `num_generations=2`, and `max_completion_length=768`.

Do not load a second Qwen2.5-3B model for probe scoring. The RLFR reward code reuses the policy model under `torch.no_grad()` to extract the probe feature.

If the pilot OOMs, lower `num_generations` before lowering `max_completion_length`.

## Interpretation

RLFR is promising only if, compared with both the hacking start checkpoint and the lambda0 continued-GRPO control:

- Task3 `mean_trace_score`, `trace_label_rate`, shortcut rate, or probe hacking score drop;
- accuracy remains acceptable;
- parser success and truncation remain sane.

Current Task3 RealTrace v0 is `real_trace_v0_prefix_ablation_3prefix`, a lightweight TRACE-style scorer, not a full TRACE reproduction.
