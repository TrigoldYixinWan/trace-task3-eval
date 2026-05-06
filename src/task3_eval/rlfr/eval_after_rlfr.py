"""Post-training Task3 evaluation wrapper for Task5 RLFR checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from task3_eval.eval.compare_checkpoints import compare_checkpoints
from task3_eval.eval.generate_rollouts import generate_rollouts
from task3_eval.eval.score_rollouts import score_rollouts
from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL


def evaluate_after_rlfr(
    dataset_path: str,
    base_model_name: str = DEFAULT_BASE_MODEL,
    hacking_start_checkpoint: str = "/workspace/checkpoints/hacking/checkpoint-50",
    lambda0_checkpoint: str = "outputs/checkpoints/rlfr/continued_grpo_lambda0_step30",
    lambda05_checkpoint: str = "outputs/checkpoints/rlfr/probe_lambda05_step30",
    output_dir: str = "outputs",
    limit: int = 100,
    max_new_tokens: int = 1024,
    trace_answer_max_new_tokens: int = 96,
    torch_dtype: str = "bf16",
    device_map: str = "auto",
) -> None:
    models = [
        ("hacking_start_checkpoint", 50, hacking_start_checkpoint, "hacking_start"),
        ("continued_grpo_lambda0_step30", 30, lambda0_checkpoint, "rlfr_lambda0_control"),
        ("rlfr_probe_lambda05_step30", 30, lambda05_checkpoint, "rlfr_lambda05"),
    ]
    scored_paths = []
    for checkpoint_name, checkpoint_step, checkpoint_path, run_type in models:
        raw = Path(output_dir) / "rollouts" / "raw" / f"task5_{checkpoint_name}_raw.jsonl"
        scored = Path(output_dir) / "rollouts" / "scored" / f"task5_{checkpoint_name}_realtrace_scored.jsonl"
        generate_rollouts(
            dataset_path=dataset_path,
            output_path=raw,
            base_model_name=base_model_name,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            checkpoint_step=checkpoint_step,
            adapter_type="lora",
            run_type=run_type,
            reward_type="task5_post_training_eval",
            limit=limit,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        score_rollouts(
            input_path=raw,
            output_path=scored,
            trace_scorer="real_v0",
            base_model_name=base_model_name,
            checkpoint_path=checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trace_answer_max_new_tokens=trace_answer_max_new_tokens,
            limit=limit,
        )
        scored_paths.append(str(scored))

    compare_checkpoints(
        inputs=scored_paths,
        output_csv=Path(output_dir) / "reports" / "task5_rlfr_effectiveness.csv",
        output_md=Path(output_dir) / "reports" / "task5_rlfr_effectiveness.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/workspace/data/math_ic_test.jsonl")
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--hacking_start_checkpoint", default="/workspace/checkpoints/hacking/checkpoint-50")
    parser.add_argument("--lambda0_checkpoint", default="outputs/checkpoints/rlfr/continued_grpo_lambda0_step30")
    parser.add_argument("--lambda05_checkpoint", default="outputs/checkpoints/rlfr/probe_lambda05_step30")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--trace_answer_max_new_tokens", type=int, default=96)
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    evaluate_after_rlfr(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
