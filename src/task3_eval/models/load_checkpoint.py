"""Load base model and optional LoRA adapter for rollout generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL = DEFAULT_BASE_MODEL


@dataclass(slots=True)
class LoadedCheckpoint:
    base_model_name: str
    checkpoint_path: str | None
    tokenizer: Any
    model: Any


def _resolve_torch_dtype(torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to resolve torch_dtype.") from exc

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError("torch_dtype must be one of: auto, fp16, bf16, fp32")
    return dtype_map[torch_dtype]


def validate_load_args(base_model_name: str, checkpoint_path: str | None, torch_dtype: str, device_map: str) -> None:
    if not base_model_name:
        raise ValueError("base_model_name is required.")
    if torch_dtype not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("torch_dtype must be one of: auto, fp16, bf16, fp32")
    if device_map not in {"auto", "cpu"}:
        raise ValueError("device_map must be one of: auto, cpu")
    if checkpoint_path == "":
        raise ValueError("checkpoint_path may be None, 'base', or a non-empty LoRA path.")


def load_model_and_tokenizer(
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    torch_dtype: str = "auto",
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> LoadedCheckpoint:
    """Load a base Hugging Face causal LM and optional PEFT LoRA adapter.

    Imports are intentionally local so smoke-test utilities can compile and run
    without importing heavyweight ML dependencies.
    """

    validate_load_args(base_model_name, checkpoint_path, torch_dtype, device_map)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for real generation. Install requirements.txt "
            "or run generation with --dry-run."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
    }
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if device_map == "cpu":
        model.to("cpu")

    if checkpoint_path not in (None, "base"):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load LoRA adapters.") from exc
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return LoadedCheckpoint(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        model=model,
    )


def load_checkpoint(
    checkpoint: str = DEFAULT_BASE_MODEL,
    lora_adapter: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    cache_dir: str | None = None,
) -> LoadedCheckpoint:
    """Backward-compatible wrapper around load_model_and_tokenizer."""

    dtype_map = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    return load_model_and_tokenizer(
        base_model_name=checkpoint,
        checkpoint_path=lora_adapter or "base",
        torch_dtype=dtype_map.get(dtype, dtype),
        device_map=device,
        cache_dir=cache_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_path", default="base")
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--cache_dir")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    validate_load_args(args.base_model_name, args.checkpoint_path, args.torch_dtype, args.device_map)
    if args.dry_run:
        print(
            "dry_run_ok "
            f"base_model_name={args.base_model_name} "
            f"checkpoint_path={args.checkpoint_path} "
            f"torch_dtype={args.torch_dtype} "
            f"device_map={args.device_map}"
        )
        return
    load_model_and_tokenizer(
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    print("load_ok")


if __name__ == "__main__":
    main()
