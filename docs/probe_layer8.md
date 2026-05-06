# Probe Layer 8 Task5 Pilot Notes

## Context

This note summarizes the Task5 RLFR pilot using the frozen probe artifact stored under:

```text
CS2952N_TRACE_Task3/probes/label_best_layer
```

The RunPod-side probe path was:

```text
/workspace/probes/label_best_layer
```

The pilot used:

```text
base_model = Qwen/Qwen2.5-3B-Instruct
start_checkpoint = /workspace/checkpoints/hacking/checkpoint-50
probe_layer_idx = 8
probe_pooling_method = completion_mean_pool
max_steps = 30
lambda_probe = 0.5
```

Evaluation reused Task3 RealTrace v0:

```text
real_trace_v0_prefix_ablation_3prefix
```

This is a lightweight TRACE-style proxy, not a full TRACE reproduction.

## Task5 Eval Summary

| model | accuracy | mean_trace_score | trace_label_rate | shortcut_rate | full_success | early_success | truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hacking_start | 0.410 | 0.1150 | 0.180 | 0.110 | 0.400 | 0.135 | 0.220 |
| lambda0_control | 0.410 | 0.1350 | 0.190 | 0.120 | 0.410 | 0.150 | 0.150 |
| lambda05_RLFR | 0.480 | 0.1375 | 0.180 | 0.130 | 0.460 | 0.170 | 0.200 |

## Interpretation

The 30-step `lambda_probe=0.5` RLFR pilot completed successfully, but it does not provide evidence that the layer-8 probe penalty reduced reward-hacking behavior under the current Task3 RealTrace-v0 evaluation.

Compared with the hacking start checkpoint, `lambda05_RLFR` improved accuracy from `0.410` to `0.480`, but `mean_trace_score`, `shortcut_rate`, and `early_success_rate` increased rather than decreased.

Compared with the `lambda0` continued-GRPO control, `lambda05_RLFR` again improved accuracy from `0.410` to `0.480`, but TRACE-style risk metrics were essentially flat or slightly worse:

```text
mean_trace_score: 0.1350 -> 0.1375
trace_label_rate: 0.190 -> 0.180
shortcut_rate: 0.120 -> 0.130
early_success_rate: 0.150 -> 0.170
```

The safest conclusion is:

```text
In this 30-step pilot, RLFR with the layer-8 frozen probe improved task accuracy, but did not reduce RealTrace-v0 shortcut indicators. The current result does not support a claim that RLFR successfully suppressed reward hacking.
```

## Quality Notes

The eval produced high `hit_max_length` warnings for:

```text
hacking_start: 0.22
lambda05_RLFR: 0.20
```

This does not invalidate the run, but it means completion truncation may still affect correctness and RealTrace-v0 measurements.

`parser_success_rate` was `1.0` for all three models, while `has_answer_tag_rate` was `0.0`, so answer extraction depended on the numeric fallback parser rather than explicit `<answer>...</answer>` tags.

## Recommended Next Checks

Before changing training length or lambda, inspect the training reward logs to verify that the probe supplied a meaningful online penalty:

```bash
python - <<'PY'
import json, statistics

for path in [
    "outputs/rlfr_logs/rlfr_continued_grpo_lambda0_reward_breakdown.jsonl",
    "outputs/rlfr_logs/rlfr_probe_lambda05_reward_breakdown.jsonl",
]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    p_hack = [float(row["p_hack"]) for row in rows]
    reward = [float(row["total_reward"]) for row in rows]
    print(path)
    print("n", len(rows))
    print("p_hack mean/min/max", statistics.mean(p_hack), min(p_hack), max(p_hack))
    print("reward mean/min/max", statistics.mean(reward), min(reward), max(reward))
PY
```

If `p_hack` is near zero or nearly constant, first verify that the online feature definition exactly matches probe training:

```text
layer_idx = 8
pooling_method = completion_mean_pool
feature dimension
normalization
activation-only vs hybrid probe
```

If `p_hack` has meaningful variation, a reasonable next pilot is to keep `max_steps=30` and test a stronger penalty such as:

```text
lambda_probe = 1.0
```

Do not claim RLFR effectiveness until Task3 post-training eval shows reduced TRACE/probe hacking metrics relative to both the hacking start checkpoint and the lambda0 continued-GRPO control while preserving acceptable accuracy.
