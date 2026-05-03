"""Load base model and optional LoRA adapter for rollout generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task3_eval.utils.cli import parse_bool


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MODEL = DEFAULT_BASE_MODEL
PEFT_CONFIG_NAME = "adapter_config.json"
PEFT_MODEL_NAME = "adapter_model.safetensors"


@dataclass(slots=True)
class LoadedCheckpoint:
    base_model_name: str
    checkpoint_path: str | None
    checkpoint_type: str
    peft_used: bool
    tokenizer: Any
    model: Any


def _is_base_checkpoint(checkpoint_path: str | None) -> bool:
    return checkpoint_path in (None, "", "base")


def _tokenizer_files_exist(path: str | Path) -> bool:
    checkpoint_dir = Path(path)
    return any(
        (checkpoint_dir / filename).exists()
        for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )


def _appears_peft_lora(path: Path) -> bool:
    return (path / PEFT_CONFIG_NAME).exists() or (path / PEFT_MODEL_NAME).exists()


def verify_peft_lora_checkpoint(checkpoint_path: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_path)
    missing = [
        filename
        for filename in (PEFT_CONFIG_NAME, PEFT_MODEL_NAME)
        if not (checkpoint_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "PEFT/LoRA checkpoint is incomplete. Missing "
            f"{', '.join(missing)} in {checkpoint_dir}"
        )
    try:
        config = json.loads((checkpoint_dir / PEFT_CONFIG_NAME).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {checkpoint_dir / PEFT_CONFIG_NAME}") from exc
    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type and peft_type != "LORA":
        raise ValueError(f"Expected peft_type=LORA, found peft_type={peft_type}")


def detect_checkpoint_type(checkpoint_path: str | None) -> str:
    """Detect whether a checkpoint path is base, PEFT LoRA, full model, or unknown."""

    if _is_base_checkpoint(checkpoint_path):
        return "base"

    path = Path(str(checkpoint_path))
    if not path.exists():
        return "unknown"
    if path.is_dir():
        if (path / PEFT_CONFIG_NAME).exists() and (path / PEFT_MODEL_NAME).exists():
            return "peft_lora"
        if _appears_peft_lora(path):
            return "unknown"
        full_model_markers = (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if any((path / marker).exists() for marker in full_model_markers):
            return "full_model"
    return "unknown"


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


def validate_load_args(
    base_model_name: str,
    checkpoint_path: str | None,
    torch_dtype: str,
    device_map: str,
    load_in_4bit: bool = False,
) -> None:
    if not base_model_name:
        raise ValueError("base_model_name is required.")
    if torch_dtype not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("torch_dtype must be one of: auto, fp16, bf16, fp32")
    if device_map not in {"auto", "cpu"}:
        raise ValueError("device_map must be one of: auto, cpu")
    if checkpoint_path == "":
        raise ValueError("checkpoint_path may be None, 'base', or a non-empty LoRA path.")
    if load_in_4bit and device_map == "cpu":
        raise ValueError("load_in_4bit is not supported with device_map=cpu")


def inspect_checkpoint(checkpoint_path: str | None) -> str:
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    if not _is_base_checkpoint(checkpoint_path):
        path = Path(str(checkpoint_path))
        if path.exists() and path.is_dir() and _appears_peft_lora(path):
            verify_peft_lora_checkpoint(path)
    return checkpoint_type


def _diagnostics(
    base_model_name: str,
    checkpoint_path: str | None,
    checkpoint_type: str,
    torch_dtype: str,
    device_map: str,
    load_in_4bit: bool,
) -> dict[str, str | bool | None]:
    return {
        "checkpoint_type": checkpoint_type,
        "checkpoint_path": checkpoint_path or "base",
        "base_model_name_used": base_model_name,
        "peft_used": checkpoint_type == "peft_lora",
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "load_in_4bit": load_in_4bit,
    }


def print_diagnostics(diagnostics: dict[str, str | bool | None]) -> None:
    for key, value in diagnostics.items():
        print(f"{key}={value}")


def _model_kwargs(torch_dtype: str, device_map: str, cache_dir: str | None, load_in_4bit: bool) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
    }
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("transformers BitsAndBytesConfig is required for load_in_4bit.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    return model_kwargs


def _resize_embeddings_if_needed(model: Any, tokenizer: Any) -> None:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    resize_embeddings = getattr(model, "resize_token_embeddings", None)
    if not callable(get_embeddings) or not callable(resize_embeddings):
        return
    embeddings = get_embeddings()
    current_size = getattr(embeddings, "num_embeddings", None)
    tokenizer_size = len(tokenizer)
    if current_size is not None and current_size != tokenizer_size:
        resize_embeddings(tokenizer_size)


def load_model_and_tokenizer(
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    cache_dir: str | None = None,
    verbose: bool = True,
) -> LoadedCheckpoint:
    """Load a base Hugging Face causal LM and optional PEFT LoRA adapter.

    Imports are intentionally local so smoke-test utilities can compile and run
    without importing heavyweight ML dependencies.
    """

    validate_load_args(base_model_name, checkpoint_path, torch_dtype, device_map, load_in_4bit)
    checkpoint_type = inspect_checkpoint(checkpoint_path)
    diagnostics = _diagnostics(base_model_name, checkpoint_path, checkpoint_type, torch_dtype, device_map, load_in_4bit)
    if verbose:
        print_diagnostics(diagnostics)
    if checkpoint_type == "unknown" and not _is_base_checkpoint(checkpoint_path):
        raise ValueError(f"Unknown or unsupported checkpoint format: {checkpoint_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for real generation. Install requirements.txt "
            "or run generation with --dry-run."
        ) from exc

    if checkpoint_type == "full_model":
        model_source = str(checkpoint_path)
        tokenizer_source = model_source if _tokenizer_files_exist(model_source) else base_model_name
    else:
        model_source = base_model_name
        tokenizer_source = str(checkpoint_path) if checkpoint_type == "peft_lora" and _tokenizer_files_exist(str(checkpoint_path)) else base_model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **_model_kwargs(torch_dtype, device_map, cache_dir, load_in_4bit),
    )
    if device_map == "cpu":
        model.to("cpu")

    if checkpoint_type == "peft_lora":
        _resize_embeddings_if_needed(model, tokenizer)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load LoRA adapters.") from exc
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return LoadedCheckpoint(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_type=checkpoint_type,
        peft_used=checkpoint_type == "peft_lora",
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
    parser.add_argument("--load_in_4bit", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--cache_dir")
    parser.add_argument("--dry_run", "--dry-run", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    validate_load_args(
        args.base_model_name,
        args.checkpoint_path,
        args.torch_dtype,
        args.device_map,
        args.load_in_4bit,
    )
    checkpoint_type = inspect_checkpoint(args.checkpoint_path)
    print_diagnostics(
        _diagnostics(
            args.base_model_name,
            args.checkpoint_path,
            checkpoint_type,
            args.torch_dtype,
            args.device_map,
            args.load_in_4bit,
        )
    )
    if args.dry_run:
        print("dry_run_ok")
        return
    load_model_and_tokenizer(
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.cache_dir,
        verbose=False,
    )
    print("load_ok")


if __name__ == "__main__":
    main()
