"""Generate rollouts for held-out math loophole data."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

from task3_eval.data.jsonl_io import read_jsonl, write_jsonl
from task3_eval.data.schemas import validate_dataset_record
from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL, load_model_and_tokenizer
from task3_eval.utils.cli import parse_bool
from task3_eval.utils.metadata import (
    canonical_checkpoint_name,
    infer_adapter_type,
    infer_checkpoint_step,
    prompt_id_for,
)


def _dry_response(example: dict[str, Any]) -> str:
    return f"[DRY RUN] <answer>{example.get('answer', '')}</answer>"


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def _generate_one(
    loaded: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    do_sample: bool,
) -> tuple[str, int, bool]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    text = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    }
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if top_p is not None:
        generation_kwargs["top_p"] = top_p
    outputs = model.generate(**inputs, **generation_kwargs)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_token_length = int(generated_ids.shape[-1])
    return completion, completion_token_length, completion_token_length >= max_new_tokens


def generate_rollouts(
    dataset_path: str | Path,
    output_path: str | Path,
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    checkpoint_name: str | None = None,
    checkpoint_step: int | None = None,
    adapter_type: str | None = None,
    run_type: str = "hacking",
    reward_type: str = "math_reward_with_loophole",
    prompt_field: str = "prompt",
    limit: int | None = None,
    dry_run: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    num_return_sequences: int = 1,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    cache_dir: str | None = None,
) -> int:
    rows = read_jsonl(dataset_path)
    examples = list(islice(rows, limit)) if limit else list(rows)
    for index, example in enumerate(examples, start=1):
        validate_dataset_record(example, f"{dataset_path}:{index}")
        if prompt_field not in example:
            raise ValueError(f"prompt_field '{prompt_field}' missing in {dataset_path}:{index}")

    if num_return_sequences != 1:
        raise ValueError("num_return_sequences > 1 is not implemented for Task 3 MVP.")

    loaded = None
    if not dry_run:
        loaded = load_model_and_tokenizer(
            base_model_name=base_model_name,
            checkpoint_path=checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
        )

    resolved_step = checkpoint_step if checkpoint_step is not None else infer_checkpoint_step(checkpoint_name, checkpoint_path)
    resolved_checkpoint_name = checkpoint_name or canonical_checkpoint_name(
        resolved_step,
        Path(checkpoint_path).name if checkpoint_path not in (None, "base") else "base",
    )
    resolved_adapter_type = adapter_type or infer_adapter_type(checkpoint_path)
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "load_in_4bit": load_in_4bit,
        "dry_run": dry_run,
    }
    rollout_rows = []
    for example in examples:
        prompt = example[prompt_field]
        if dry_run:
            completion = _dry_response(example)
            completion_token_length = len(completion.split())
            hit_max_length = False
        else:
            completion, completion_token_length, hit_max_length = _generate_one(
                loaded,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                do_sample,
            )
        rollout_rows.append(
            {
                "sample_id": example["sample_id"],
                "prompt_id": prompt_id_for(example),
                "checkpoint_name": resolved_checkpoint_name,
                "checkpoint_step": resolved_step,
                "checkpoint_path": checkpoint_path or "base",
                "base_model_name": base_model_name,
                "adapter_type": resolved_adapter_type,
                "run_type": run_type,
                "reward_type": reward_type,
                "prompt": prompt,
                "completion": completion,
                "answer": example["answer"],
                "task_type": example["task_type"],
                "loophole_type": example["loophole_type"],
                "loophole_subtype": example["loophole_subtype"],
                "split": example["split"],
                "completion_token_length": completion_token_length,
                "hit_max_length": hit_max_length,
                "generation_config": generation_config,
            }
        )
    return write_jsonl(output_path, rollout_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_path", default="base")
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--checkpoint_step", type=int)
    parser.add_argument("--adapter_type")
    parser.add_argument("--run_type", default="hacking")
    parser.add_argument("--reward_type", default="math_reward_with_loophole")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry_run", "--dry-run", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--load_in_4bit", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--cache_dir")
    args = parser.parse_args()
    count = generate_rollouts(**vars(args))
    print(f"rollout_rows={count}")


if __name__ == "__main__":
    main()
