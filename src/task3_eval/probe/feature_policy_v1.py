"""Feature Policy v1 extraction for Task 3 probe artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from task3_eval.data.jsonl_io import read_jsonl
from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL, load_model_and_tokenizer
from task3_eval.utils.artifact_budget import budget_warnings, optional_feature_budget_available
from task3_eval.utils.cli import parse_bool


POLICY_POOLED_LAYERS = [12, 20, 30, 35]
POLICY_POOLED_METHODS = ["completion_last_token", "completion_mean_pool"]
POLICY_ALL_TOKEN_LAYERS = [20, 30]
POLICY_ALL_TOKEN_STEPS = {10, 70, 130}
POLICY_MAX_SEQ_LEN = 512


def _rows_for_checkpoint(
    probe_dataset_path: str | Path,
    checkpoint_step: int | None,
    run_type: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = list(read_jsonl(probe_dataset_path))
    if checkpoint_step is not None:
        rows = [row for row in rows if int(row.get("checkpoint_step") or -1) == checkpoint_step]
    if run_type:
        rows = [row for row in rows if row.get("run_type") == run_type]
    rows = [row for row in rows if row.get("label_for_probe") is not None]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No labeled probe dataset rows matched the requested filters.")
    return rows


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def _completion_bounds(tokenizer: Any, row: dict[str, Any], input_ids: Any, attention_mask: Any) -> tuple[int, int]:
    prompt_ids = tokenizer(str(row.get("prompt", "")), add_special_tokens=False)["input_ids"]
    seq_len = int(attention_mask.sum().item())
    start = min(len(prompt_ids), max(seq_len - 1, 0))
    end = max(seq_len, start + 1)
    return start, end


def _pool_layer(layer_hidden: Any, start: int, end: int) -> list[Any]:
    completion_hidden = layer_hidden[0, start:end, :]
    return [
        completion_hidden[-1, :],
        completion_hidden.mean(dim=0),
    ]


def _metadata_lists(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = (
        "sample_id",
        "prompt_id",
        "checkpoint_name",
        "checkpoint_step",
        "run_type",
        "reward_type",
        "label_for_probe",
        "label_source",
    )
    return {field: [row.get(field) for row in rows] for field in fields}


def save_pooled_features(
    probe_dataset_path: str | Path = "outputs/probe_dataset/task3_probe_dataset.jsonl",
    output_dir: str | Path = "outputs/probe_features/pooled",
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    checkpoint_step: int | None = None,
    run_type: str = "hacking",
    layers: list[int] | None = None,
    pooling_methods: list[str] | None = None,
    max_seq_len: int = 1024,
    limit: int | None = None,
    torch_dtype: str = "bf16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    output_root: str | Path = "outputs",
) -> str:
    import torch

    selected_layers = layers or POLICY_POOLED_LAYERS
    selected_pooling = pooling_methods or POLICY_POOLED_METHODS
    if selected_pooling != POLICY_POOLED_METHODS:
        raise ValueError(f"Feature Policy v1 pooled methods must be {POLICY_POOLED_METHODS}")
    rows = _rows_for_checkpoint(probe_dataset_path, checkpoint_step, run_type, limit)
    loaded = load_model_and_tokenizer(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = loaded.tokenizer
    model = loaded.model
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_rows = []
    with torch.no_grad():
        for row in rows:
            text = f"{row.get('prompt', '')}\n\n{row.get('completion', '')}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            inputs = _move_inputs_to_model(inputs, model)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            start, end = _completion_bounds(tokenizer, row, inputs["input_ids"][0], inputs["attention_mask"][0])
            layer_features = []
            for layer_id in selected_layers:
                pooled = _pool_layer(outputs.hidden_states[layer_id], start, end)
                layer_features.append(torch.stack(pooled).detach().to(torch.float16).cpu())
            feature_rows.append(torch.stack(layer_features))
    feature_tensor = torch.stack(feature_rows)
    output_path = Path(output_dir) / f"{run_type}_checkpoint-{checkpoint_step}_pooled_features.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **_metadata_lists(rows),
            "layers": selected_layers,
            "pooling_methods": selected_pooling,
            "features": feature_tensor,
            "label_source_warning": "heuristic_trace_v0 proxy labels, not real TRACE",
        },
        output_path,
    )
    for warning in budget_warnings(output_root):
        print(f"WARNING: {warning}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output_path)


def save_selected_all_token_features(
    probe_dataset_path: str | Path = "outputs/probe_dataset/task3_probe_dataset.jsonl",
    output_dir: str | Path = "outputs/probe_features/all_token_selected",
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    checkpoint_step: int | None = None,
    run_type: str = "hacking",
    max_seq_len: int = POLICY_MAX_SEQ_LEN,
    limit: int | None = None,
    torch_dtype: str = "bf16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    output_root: str | Path = "outputs",
) -> str | None:
    import torch

    if checkpoint_step not in POLICY_ALL_TOKEN_STEPS:
        print(f"Skipping all-token features for checkpoint-{checkpoint_step}; policy allows {sorted(POLICY_ALL_TOKEN_STEPS)}")
        return None
    if not optional_feature_budget_available(output_root):
        print("WARNING: output budget approaching limit; skipping optional all-token feature extraction")
        return None

    rows = _rows_for_checkpoint(probe_dataset_path, checkpoint_step, run_type, limit)
    loaded = load_model_and_tokenizer(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = loaded.tokenizer
    model = loaded.model
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_rows = []
    attention_masks = []
    token_positions = []
    with torch.no_grad():
        for row in rows:
            text = f"{row.get('prompt', '')}\n\n{row.get('completion', '')}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len, padding="max_length")
            inputs = _move_inputs_to_model(inputs, model)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            layer_features = [
                outputs.hidden_states[layer_id][0, :max_seq_len, :].detach().to(torch.float16).cpu()
                for layer_id in POLICY_ALL_TOKEN_LAYERS
            ]
            feature_rows.append(torch.stack(layer_features))
            attention_masks.append(inputs["attention_mask"][0].detach().cpu().to(torch.bool))
            token_positions.append(torch.arange(max_seq_len, dtype=torch.int16))
    output_path = Path(output_dir) / f"{run_type}_checkpoint-{checkpoint_step}_layers20_30_alltoken.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **_metadata_lists(rows),
            "layers": POLICY_ALL_TOKEN_LAYERS,
            "max_seq_len": max_seq_len,
            "features": torch.stack(feature_rows),
            "attention_mask": torch.stack(attention_masks),
            "token_positions": torch.stack(token_positions),
            "label_source_warning": "heuristic_trace_v0 proxy labels, not real TRACE",
        },
        output_path,
    )
    for warning in budget_warnings(output_root):
        print(f"WARNING: {warning}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output_path)


def _parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pooled", "all_token_selected"], required=True)
    parser.add_argument("--probe_dataset_path", default="outputs/probe_dataset/task3_probe_dataset.jsonl")
    parser.add_argument("--output_dir")
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_path", default="base")
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--run_type", default="hacking")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in POLICY_POOLED_LAYERS))
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--load_in_4bit", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    if args.mode == "pooled":
        path = save_pooled_features(
            probe_dataset_path=args.probe_dataset_path,
            output_dir=args.output_dir or "outputs/probe_features/pooled",
            base_model_name=args.base_model_name,
            checkpoint_path=args.checkpoint_path,
            checkpoint_step=args.checkpoint_step,
            run_type=args.run_type,
            layers=_parse_layers(args.layers),
            max_seq_len=args.max_seq_len,
            limit=args.limit,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        path = save_selected_all_token_features(
            probe_dataset_path=args.probe_dataset_path,
            output_dir=args.output_dir or "outputs/probe_features/all_token_selected",
            base_model_name=args.base_model_name,
            checkpoint_path=args.checkpoint_path,
            checkpoint_step=args.checkpoint_step,
            run_type=args.run_type,
            max_seq_len=POLICY_MAX_SEQ_LEN,
            limit=args.limit,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
        )
    print(f"feature_output={path}")


if __name__ == "__main__":
    main()
