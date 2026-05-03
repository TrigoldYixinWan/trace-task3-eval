# RunPod Setup and Google Drive Sync

This guide describes how to run the Task 3 evaluation MVP on RunPod and copy outputs to Google Drive with `rclone`.

Task 3 is evaluation only. The setup below creates a Python environment, cache directories, data/checkpoint/output directories, and leaves all credentials outside the repository.

## Clone the Repo

On the RunPod instance, clone the repository under `/workspace`:

```bash
cd /workspace
git clone <YOUR_REPO_URL> trace-task3-eval
cd /workspace/trace-task3-eval
```

The scripts assume this default path:

```text
/workspace/trace-task3-eval
```

If needed, override it when running setup:

```bash
REPO_DIR=/workspace/trace-task3-eval bash scripts/setup_runpod.sh
```

## Run Setup

Run:

```bash
bash scripts/setup_runpod.sh
```

The script creates:

```text
/workspace/venvs/trace-task3
/workspace/data
/workspace/checkpoints
/workspace/outputs
/workspace/hf_cache
```

It also installs `requirements.txt` and installs this repo in editable mode without writing any secrets or tokens.

Activate the environment after setup:

```bash
cd /workspace/trace-task3-eval
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
source /workspace/venvs/trace-task3/bin/activate
```

## Configure Rclone Google Drive Remote

Install `rclone` on the RunPod image if it is not already available. Then configure a Google Drive remote named `gdrive`:

```bash
rclone config
```

Choose the Google Drive backend and complete the OAuth flow according to your environment.

Important: Google credentials, OAuth tokens, and `rclone.conf` must not be committed to this repository. Keep them in the RunPod user config location or another external secret store.

You can verify the remote with:

```bash
rclone lsd gdrive:
```

## Run Task 3 Smoke Test

After activating the environment:

```bash
python -m task3_eval.data.build_math_fixture \
  --output /workspace/outputs/math_ic_smoke.jsonl \
  --n 5

python -m task3_eval.eval.generate_rollouts \
  --dataset_path /workspace/outputs/math_ic_smoke.jsonl \
  --output_path /workspace/outputs/task3_smoke_rollouts.jsonl \
  --base_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint_path base \
  --checkpoint_name base \
  --limit 5 \
  --dry_run

python -m task3_eval.eval.score_rollouts \
  --input_path /workspace/outputs/task3_smoke_rollouts.jsonl \
  --output_path /workspace/outputs/task3_smoke_scored.jsonl \
  --report_json /workspace/outputs/task3_smoke_report.json
```

This smoke test uses dry-run generation, so it validates the data and scoring pipeline without downloading model weights.

## Sync Outputs to Google Drive

By default, the sync script uses `rclone copy`, not `rclone sync`, so it will not delete remote files:

```bash
bash scripts/sync_to_gdrive.sh
```

Defaults:

```text
local dir: outputs/
remote:    gdrive:CS2952N_TRACE_Task3
mode:      copy
```

Override the remote path:

```bash
bash scripts/sync_to_gdrive.sh \
  --local_dir /workspace/outputs \
  --remote gdrive:CS2952N_TRACE_Task3/runpod_smoke \
  --mode copy
```

Use `--mode sync` only when you intentionally want remote files deleted if they are absent locally:

```bash
bash scripts/sync_to_gdrive.sh \
  --local_dir /workspace/outputs \
  --remote gdrive:CS2952N_TRACE_Task3/latest \
  --mode sync
```

The script prints the `rclone` command before executing it and fails clearly if `rclone` is missing.
