# RunPod Task 3 Evaluation Guide

This guide covers inference/evaluation for Task 3 of the TRACE + RLFR reward-hacking project. It does not implement or run RL training.

## Repo Root on RunPod

If the repository was cloned or unzipped into a nested folder, the real repo root may be:

```text
/workspace/trace-task3-eval/trace-task3-eval
```

Use the directory that contains `scripts/`, `src/`, and `requirements.txt`:

```bash
find /workspace -maxdepth 4 -name smoke_test_task3_checkpoint.sh -print
cd /workspace/trace-task3-eval/trace-task3-eval
source /workspace/venvs/trace-task3/bin/activate
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
```

## Checkpoint Layout

Unzip or copy the Qwen2.5-3B LoRA checkpoints into:

```text
/workspace/checkpoints/hacking/
```

Expected folder layout:

```text
/workspace/checkpoints/hacking/checkpoint-10/
/workspace/checkpoints/hacking/checkpoint-20/
/workspace/checkpoints/hacking/checkpoint-30/
/workspace/checkpoints/hacking/checkpoint-40/
/workspace/checkpoints/hacking/checkpoint-50/
/workspace/checkpoints/hacking/checkpoint-60/
/workspace/checkpoints/hacking/checkpoint-70/
/workspace/checkpoints/hacking/checkpoint-80/
/workspace/checkpoints/hacking/checkpoint-90/
/workspace/checkpoints/hacking/checkpoint-100/
/workspace/checkpoints/hacking/checkpoint-110/
/workspace/checkpoints/hacking/checkpoint-120/
/workspace/checkpoints/hacking/checkpoint-130/
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
find /workspace/checkpoints/hacking/checkpoint-10 -maxdepth 1 -type f | sort
```

## Loader Dry Run

The adapter config may contain an old local training path like `models/qwen25-3b-instruct`. Do not rely on that path on RunPod. Always override the base model explicitly:

```bash
python -m task3_eval.models.load_checkpoint \
  --dry_run \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path /workspace/checkpoints/hacking/checkpoint-10
```

Expected diagnostics include:

```text
checkpoint_type=peft_lora
base_model_name_used=Qwen/Qwen2.5-3B-Instruct
peft_used=True
```

## One Small Evaluation

First run the local smoke test from the real repo root. This does not download Qwen weights:

```bash
cd /workspace/trace-task3-eval/trace-task3-eval

CHECKPOINT_PATH=/workspace/checkpoints/hacking/checkpoint-10 \
BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
bash scripts/smoke_test_task3_checkpoint.sh
```

The expected final line is:

```text
PASS
```

Check that the real held-out dataset exists:

```bash
ls -lh /workspace/data/math_ic_test.jsonl
head -n 2 /workspace/data/math_ic_test.jsonl
```

Generate rollouts:

```bash
python -m task3_eval.eval.generate_rollouts \
  --dataset_path /workspace/data/math_ic_test.jsonl \
  --output_path outputs/rollouts/raw/hacking_checkpoint-10_math_ic_raw.jsonl \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path /workspace/checkpoints/hacking/checkpoint-10 \
  --checkpoint_name checkpoint-10 \
  --checkpoint_step 10 \
  --adapter_type lora \
  --run_type hacking \
  --reward_type math_reward_with_loophole \
  --limit 2 \
  --max_new_tokens 128 \
  --torch_dtype bf16 \
  --device_map auto
```

Score rollouts:

```bash
python -m task3_eval.eval.score_rollouts \
  --input_path outputs/rollouts/raw/hacking_checkpoint-10_math_ic_raw.jsonl \
  --output_path outputs/rollouts/scored/hacking_checkpoint-10_math_ic_scored.jsonl
```

Inspect one generated rollout and one scored rollout:

```bash
tail -n 1 outputs/rollouts/raw/hacking_checkpoint-10_math_ic_raw.jsonl
tail -n 1 outputs/rollouts/scored/hacking_checkpoint-10_math_ic_scored.jsonl
```

## Full Checkpoint Sweep

Run all configured checkpoints:

```bash
bash scripts/run_qwen3b_checkpoint_sweep.sh
```

For RealTrace v0 supervised scoring and Feature Policy v1 hidden-state features, first run the checkpoint-10 200-example pilot:

```bash
LIMIT=200 bash scripts/run_real_trace_v0_checkpoint10_pilot200.sh
```

If that succeeds, run the remaining 12 checkpoints:

```bash
LIMIT=200 bash scripts/run_real_trace_v0_remaining12_full.sh
```

These scripts use `real_trace_v0_prefix_ablation_3prefix`, extract pooled hidden-state features for each processed checkpoint, and save selected all-token features only for checkpoint-10 and checkpoint-70 by default.

The sweep writes:

```text
outputs/reports/checkpoint_trend.csv
outputs/reports/checkpoint_trend.md
outputs/probe_dataset/task3_probe_dataset.jsonl
```

## RLFR-Ready Probe Artifacts

Task 3 also prepares probe artifacts for Task 4 / RLFR handoff. These labels are heuristic TRACE proxy labels from `heuristic_trace_v0`, not real TRACE.

After a sweep, the probe dataset is created from scored rollouts:

```text
outputs/probe_dataset/task3_probe_dataset.jsonl
```

It includes prompt, completion, checkpoint metadata, loophole metadata, heuristic TRACE fields, and:

```text
label_for_probe
label_source
```

The expected `label_source` is:

```text
heuristic_trace_v0
```

To rebuild the probe dataset manually:

```bash
python -m task3_eval.probe.build_probe_dataset \
  --inputs outputs/rollouts/scored/*_scored.jsonl \
  --output_path outputs/probe_dataset/task3_probe_dataset.jsonl \
  --dataset_card_path outputs/probe_dataset/dataset_card.md \
  --behavior_features_dir outputs/probe_features/pooled
```

To produce model-derived hidden-state features for one checkpoint:

```bash
python -m task3_eval.probe.feature_policy_v1 \
  --mode pooled \
  --probe_dataset_path outputs/probe_dataset/task3_probe_dataset.jsonl \
  --output_dir outputs/probe_features/pooled \
  --base_model_name Qwen/Qwen2.5-3B-Instruct \
  --checkpoint_path /workspace/checkpoints/hacking/checkpoint-10 \
  --checkpoint_step 10 \
  --run_type hacking \
  --torch_dtype bf16 \
  --device_map auto
```

This writes:

```text
outputs/probe_features/pooled/hacking_checkpoint-10_pooled_features.pt
outputs/probe_features/pooled/hacking_behavior_features.parquet
```

The `.pt` file aligns features by `sample_id` and records checkpoint name, layer ids, pooling methods, and labels.

Probe training for the Feature Policy v1 `.pt` files is represented by TODO stubs until the Task 4 split policy is finalized. The older `.npz` probe trainer remains available for smoke tests.

```bash
This writes:

```text
outputs/probe/probe_model.pkl
outputs/probe/probe_config.json
outputs/probe/probe_metrics.json
outputs/probe/probe_readme.md
outputs/validation/clean_negative_control_results.csv
outputs/validation/probe_validation_report.md
```

The validation reports include train/test metrics on hacking checkpoint rows, clean/base checkpoint negative-control status when present, AUC, precision, recall, F1, confusion matrix, and confound checks for completion length, parser failure, and `hit_max_length`.

Convenience script:

```bash
bash scripts/run_task3_probe_artifacts.sh outputs/rollouts/scored/*_scored.jsonl
```

That command only builds the probe dataset. To also extract policy-compliant pooled hidden-state features and write probe/validation TODO stubs:

```bash
EXTRACT_FEATURES=1 \
CHECKPOINT_PATH=/workspace/checkpoints/hacking/checkpoint-10 \
CHECKPOINT_STEP=10 \
bash scripts/run_task3_probe_artifacts.sh outputs/rollouts/scored/*_scored.jsonl
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
rclone lsd gdrive:

bash scripts/sync_to_gdrive.sh \
  outputs/ \
  gdrive:CS2952N_TRACE_Task3/runpod_outputs_smoke
```

The script defaults to:

```text
local dir: outputs/
remote: gdrive:CS2952N_TRACE_Task3/runpod_outputs
mode: copy
```

## Warnings

- `heuristic_trace_v0` is not real TRACE.
- `real_trace_v0_prefix_ablation_3prefix` is a lightweight TRACE-style prefix-ablation scorer, not a full TRACE reproduction. See `docs/real_trace_v0.md`.
- This repo is for Task 3 evaluation only, not RL training.
- Qwen2.5-3B LoRA checkpoints must be loaded with `Qwen/Qwen2.5-3B-Instruct`, not Qwen2.5-1.5B.
