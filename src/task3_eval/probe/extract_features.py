"""Extract hidden-state features for Task 3 probe training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task3_eval.data.jsonl_io import read_jsonl, write_jsonl
from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL, load_model_and_tokenizer
from task3_eval.utils.cli import parse_bool


DEFAULT_OUTPUT_DIR = "outputs/probe_features"


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def _feature_text(row: dict[str, Any]) -> str:
    return f"{row.get('prompt', '')}\n\n{row.get('completion', '')}"


def _pool_hidden_state(hidden_state: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        return hidden_state[0, lengths[0], :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(dim=1)[0] / mask.sum(dim=1)[0].clamp(min=1)
    raise ValueError("pooling must be one of: last, mean")


def _extract_one(
    loaded: Any,
    row: dict[str, Any],
    layer_id: int,
    pooling: str,
    max_length: int,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for hidden-state feature extraction.") from exc

    tokenizer = loaded.tokenizer
    model = loaded.model
    inputs = tokenizer(
        _feature_text(row),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = _move_inputs_to_model(inputs, model)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    hidden_state = hidden_states[layer_id]
    pooled = _pool_hidden_state(hidden_state, inputs["attention_mask"], pooling)
    return pooled.detach().float().cpu().numpy()


def extract_features(
    probe_dataset_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    checkpoint_name: str | None = None,
    checkpoint_name_filter: str | None = None,
    layer_id: int = -1,
    pooling: str = "mean",
    max_length: int = 1024,
    limit: int | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    cache_dir: str | None = None,
) -> dict[str, str | int]:
    rows = list(read_jsonl(probe_dataset_path))
    if checkpoint_name_filter:
        rows = [row for row in rows if row.get("checkpoint_name") == checkpoint_name_filter]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found in probe dataset: {probe_dataset_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loaded = load_model_and_tokenizer(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
    )

    features = []
    manifest_rows = []
    for row in rows:
        feature = _extract_one(loaded, row, layer_id, pooling, max_length)
        features.append(feature)
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "checkpoint_name": checkpoint_name or row.get("checkpoint_name"),
                "checkpoint_path": checkpoint_path or row.get("checkpoint_path"),
                "base_model_name": base_model_name,
                "layer_id": layer_id,
                "token_position": None,
                "pooling_method": pooling,
                "label": int(row["label_for_probe"]),
                "label_source": row.get("label_source", "heuristic_trace_v0"),
                "completion_token_length": row.get("completion_token_length"),
                "parser_success": row.get("parser_success"),
                "hit_max_length": row.get("hit_max_length"),
                "loophole_type": row.get("loophole_type"),
                "loophole_subtype": row.get("loophole_subtype"),
            }
        )

    feature_matrix = np.vstack(features)
    labels = np.array([row["label"] for row in manifest_rows], dtype=np.int64)
    sample_ids = np.array([row["sample_id"] for row in manifest_rows])
    npz_path = output_path / "features.npz"
    manifest_path = output_path / "manifest.jsonl"
    config_path = output_path / "feature_config.json"
    np.savez_compressed(npz_path, X=feature_matrix, y=labels, sample_id=sample_ids)
    write_jsonl(manifest_path, manifest_rows)
    config = {
        "feature_type": "model_hidden_state",
        "probe_dataset_path": str(probe_dataset_path),
        "base_model_name": base_model_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "checkpoint_name_filter": checkpoint_name_filter,
        "layer_id": layer_id,
        "pooling_method": pooling,
        "max_length": max_length,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "load_in_4bit": load_in_4bit,
        "label_source_warning": "heuristic_trace_v0 proxy labels, not real TRACE",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "rows": len(rows),
        "features_npz": str(npz_path),
        "manifest_jsonl": str(manifest_path),
        "feature_config_json": str(config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dataset_path", default="outputs/probe_dataset/task3_probe_dataset.jsonl")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_path", default="base")
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--checkpoint_name_filter")
    parser.add_argument("--layer_id", type=int, default=-1)
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--load_in_4bit", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--cache_dir")
    result = extract_features(**vars(parser.parse_args()))
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
