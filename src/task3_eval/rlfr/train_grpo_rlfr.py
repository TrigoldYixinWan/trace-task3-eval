"""Task5 RLFR training entry point using TRL GRPOTrainer when available."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from task3_eval.data.jsonl_io import read_jsonl
from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL
from task3_eval.rlfr.probe_loader import FrozenProbe, load_frozen_probe
from task3_eval.rlfr.reward import compute_rlfr_reward
from task3_eval.utils.cli import parse_bool


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "rlfr_probe_lambda05",
    "base_model_name": DEFAULT_BASE_MODEL,
    "start_checkpoint_path": "/workspace/checkpoints/hacking/checkpoint-50",
    "train_dataset_path": "/workspace/data/math_ic_train.jsonl",
    "output_dir": "outputs/checkpoints/rlfr/probe_lambda05_step30",
    "probe_path": "/workspace/probes/label_best_layer",
    "probe_architecture": "linear",
    "probe_model_key": None,
    "probe_hidden_size": 2048,
    "probe_layer_idx": 8,
    "probe_layer_indices": None,
    "probe_pooling_method": "completion_mean_pool",
    "feature_normalization": None,
    "lambda_probe": 0.5,
    "max_steps": 30,
    "num_generations": 2,
    "max_completion_length": 768,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1.0e-6,
    "bf16": True,
    "gradient_checkpointing": True,
    "save_steps": 10,
    "logging_steps": 1,
    "beta": 0.001,
    "limit_train_samples": 500,
    "torch_dtype": "bf16",
    "device_map": "auto",
    "allow_dummy_probe": False,
}


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _merge_config(config_path: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    config = {**DEFAULT_CONFIG, **_load_yaml_config(config_path)}
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "\n".join(str(item) for item in completion)
    return str(completion)


def _normalize_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))
    return str(prompt)


def _metadata_at(values: Any, idx: int, default: str = "") -> Any:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if not values:
            return default
        return values[idx % len(values)]
    return values


def _load_train_records(path: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        rows.append(
            {
                "prompt": row.get("prompt", row.get("prompt_clean", "")),
                "answer": row.get("answer", ""),
                "sample_id": row.get("sample_id", ""),
                "prompt_id": row.get("prompt_id", row.get("sample_id", "")),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def _torch_dtype(torch_dtype: str) -> Any:
    import torch

    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "fp32":
        return torch.float32
    return "auto"


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _mark_only_lora_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name or "modules_to_save" in name


def _load_policy_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Task5 training requires torch, transformers, peft, and trl-related dependencies.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(str(config.get("torch_dtype", "bf16"))),
    }
    if config.get("device_map") == "auto":
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(config["base_model_name"], **model_kwargs)
    start_checkpoint = config.get("start_checkpoint_path")
    if start_checkpoint and start_checkpoint not in {"base", "none", "null"}:
        if not Path(start_checkpoint).exists():
            raise FileNotFoundError(f"start_checkpoint_path not found: {start_checkpoint}")
        try:
            model = PeftModel.from_pretrained(base_model, start_checkpoint, is_trainable=True)
        except TypeError:
            model = PeftModel.from_pretrained(base_model, start_checkpoint)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base_model, lora_config)

    _mark_only_lora_trainable(model)
    if bool(config.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters were found.")
    if torch.cuda.is_available() and config.get("device_map") != "auto":
        model.to("cuda")
    return model, tokenizer


def _bf16_enabled(requested: bool) -> bool:
    if not requested:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _build_grpo_config(config: dict[str, Any], GRPOConfig: Any) -> Any:
    raw_kwargs = {
        "output_dir": config["output_dir"],
        "run_name": config["run_name"],
        "max_steps": int(config["max_steps"]),
        "num_generations": int(config["num_generations"]),
        "max_completion_length": int(config["max_completion_length"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "bf16": _bf16_enabled(bool(config.get("bf16", True))),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "save_steps": int(config.get("save_steps", 10)),
        "logging_steps": int(config.get("logging_steps", 1)),
        "beta": float(config.get("beta", 0.001)),
        "report_to": [],
        "remove_unused_columns": False,
    }
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered = {key: value for key, value in raw_kwargs.items() if key in accepted}
    return GRPOConfig(**filtered)


def _build_reward_func(
    model: Any,
    tokenizer: Any,
    probe: FrozenProbe,
    config: dict[str, Any],
    reward_log_path: Path,
) -> Any:
    reward_log_path.parent.mkdir(parents=True, exist_ok=True)
    counter = {"value": 0}

    def reward_func(prompts, completions, **kwargs):  # type: ignore[no-untyped-def]
        answers = kwargs.get("answer")
        sample_ids = kwargs.get("sample_id")
        prompt_ids = kwargs.get("prompt_id")
        rewards = []
        with reward_log_path.open("a", encoding="utf-8") as handle:
            for idx, completion_value in enumerate(completions):
                prompt_text = _normalize_prompt(_metadata_at(prompts, idx))
                completion_text = _normalize_completion(completion_value)
                breakdown = compute_rlfr_reward(
                    prompt=prompt_text,
                    completion=completion_text,
                    answer=str(_metadata_at(answers, idx)),
                    lambda_probe=float(config["lambda_probe"]),
                    tokenizer=tokenizer,
                    model=model,
                    probe=probe,
                    fail_on_probe_error=True,
                )
                counter["value"] += 1
                log_row = {
                    "step": counter["value"],
                    "sample_id": _metadata_at(sample_ids, idx),
                    "prompt_id": _metadata_at(prompt_ids, idx),
                    **asdict(breakdown),
                }
                handle.write(json.dumps(log_row, ensure_ascii=True, sort_keys=True) + "\n")
                rewards.append(float(breakdown.total_reward))
        return rewards

    return reward_func


def print_run_summary(config: dict[str, Any]) -> None:
    print("Task5 RLFR GRPO run summary")
    print(f"run_name={config['run_name']}")
    print(f"start_checkpoint_path={config.get('start_checkpoint_path')}")
    print(f"lambda_probe={config['lambda_probe']}")
    print(f"probe_path={config.get('probe_path')}")
    print(f"probe_layer_idx={config.get('probe_layer_indices') or config['probe_layer_idx']}")
    print(f"probe_pooling_method={config['probe_pooling_method']}")
    print(f"max_steps={config['max_steps']}")
    print(f"max_completion_length={config['max_completion_length']}")
    print(f"num_generations={config['num_generations']}")
    print(f"output_dir={config['output_dir']}")


def dry_run_reward_only() -> None:
    probe = load_frozen_probe(None, layer_idx=8, pooling_method="completion_mean_pool", allow_dummy=True, verbose=False)
    examples = [
        ("What is 2 + 2?", "<answer>4</answer>", "4", 0.0),
        ("What is 2 + 2?", "<answer>4</answer>", "4", 0.5),
        ("What is 2 + 2?", "<answer>5</answer>", "4", 0.5),
    ]
    for prompt, completion, answer, lambda_probe in examples:
        breakdown = compute_rlfr_reward(
            prompt=prompt,
            completion=completion,
            answer=answer,
            lambda_probe=lambda_probe,
            probe=probe,
        )
        print(json.dumps(asdict(breakdown), sort_keys=True))
    print("dry_run_reward_only_ok")


def train_grpo_rlfr(config: dict[str, Any]) -> dict[str, str]:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("TRL with GRPOTrainer is required for real Task5 training. Install `trl`.") from exc

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir already exists and is non-empty: {output_dir}")

    print_run_summary(config)
    records = _load_train_records(config["train_dataset_path"], config.get("limit_train_samples"))
    dataset = Dataset.from_list(records)
    model, tokenizer = _load_policy_model_and_tokenizer(config)
    allow_dummy = bool(config.get("allow_dummy_probe", False)) or float(config["lambda_probe"]) == 0.0
    layer_spec = config.get("probe_layer_indices")
    if layer_spec in (None, "", "null"):
        layer_spec = config["probe_layer_idx"]
    probe = load_frozen_probe(
        probe_path=config.get("probe_path"),
        probe_architecture=config.get("probe_architecture", "linear"),
        hidden_size=int(config.get("probe_hidden_size", 2048)),
        layer_idx=layer_spec,
        pooling_method=config["probe_pooling_method"],
        feature_normalization=config.get("feature_normalization"),
        model_key=config.get("probe_model_key"),
        allow_dummy=allow_dummy,
    )
    if hasattr(probe.model, "to"):
        probe.model.to(_model_device(model))

    reward_log_path = Path("outputs/rlfr_logs") / f"{config['run_name']}_reward_breakdown.jsonl"
    reward_func = _build_reward_func(model, tokenizer, probe, config, reward_log_path)
    grpo_args = _build_grpo_config(config, GRPOConfig)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": reward_func,
        "args": grpo_args,
        "train_dataset": dataset,
    }
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {
        "output_dir": str(output_dir),
        "reward_log_path": str(reward_log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--base_model_name")
    parser.add_argument("--start_checkpoint_path")
    parser.add_argument("--train_dataset_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--probe_path")
    parser.add_argument("--probe_model_key")
    parser.add_argument("--probe_layer_idx", type=int)
    parser.add_argument("--probe_layer_indices")
    parser.add_argument("--probe_pooling_method")
    parser.add_argument("--lambda_probe", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--num_generations", type=int)
    parser.add_argument("--max_completion_length", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--bf16", nargs="?", const=True, default=None, type=parse_bool)
    parser.add_argument("--gradient_checkpointing", nargs="?", const=True, default=None, type=parse_bool)
    parser.add_argument("--limit_train_samples", type=int)
    parser.add_argument("--dry_run_reward_only", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--allow_dummy_probe", nargs="?", const=True, default=None, type=parse_bool)
    args = parser.parse_args()
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    dry_run = bool(overrides.pop("dry_run_reward_only"))
    config = _merge_config(config_path, overrides)
    if dry_run:
        dry_run_reward_only()
        return
    result = train_grpo_rlfr(config)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
