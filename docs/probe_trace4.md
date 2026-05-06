# TRACE-Supervised 4-Layer Probe Task5 Pilot

This note tracks the Task5 RLFR pilot that uses the new TRACE-supervised multi-layer probe:

```text
gdrive:CS2952N_TRACE_Task3/probes/TRACE_label_multiple_layer/probe_4_layers.pk
```

RunPod target path:

```text
/workspace/probes/TRACE_label_multiple_layer/probe_4_layers.pk
```

## Assumed Feature Definition

The default config assumes:

```text
probe_layer_indices = [12, 20, 30, 35]
probe_pooling_method = completion_mean_pool
hidden_size = 2048
expected_input_dim = 8192
```

If the loader reports `detected_input_dim=16384`, the probe may have been trained on two pooling methods, likely:

```text
completion_last_token,completion_mean_pool
```

In that case rerun the loader and training with:

```bash
PROBE_POOLING_METHOD=completion_last_token,completion_mean_pool
```

## Smoke Test

```bash
bash scripts/smoke_test_trace_probe4_loader.sh
```

This smoke test creates a tiny joblib `.pk` sklearn probe and verifies that the RLFR loader can load a 4-layer concatenated feature definition without loading Qwen.

## Training Script

```bash
PROBE_PATH=/workspace/probes/TRACE_label_multiple_layer/probe_4_layers.pk \
bash scripts/run_rlfr_pilot_trace_probe4_lambda05.sh
```

The output checkpoint defaults to:

```text
outputs/checkpoints/rlfr/trace_probe4_lambda05_step30
```

Reward logs default to:

```text
outputs/rlfr_logs/rlfr_trace_probe4_lambda05_reward_breakdown.jsonl
```
