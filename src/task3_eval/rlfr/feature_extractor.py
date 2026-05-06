"""Online hidden-state feature extraction for Task5 RLFR rewards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


SUPPORTED_POOLING = {"completion_last_token", "completion_mean_pool", "gen_first32_mean", "answer_span_mean"}


@dataclass(slots=True)
class ExtractedFeature:
    features: Any
    input_token_length: int
    completion_start: int
    completion_end: int
    layer_idx: int
    pooling_method: str


def format_prompt_for_generation(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _pool_completion(hidden: Any, completion_start: int, completion_end: int, pooling_method: str) -> Any:
    import torch

    if completion_end <= completion_start:
        completion_start = max(0, hidden.shape[1] - 1)
        completion_end = hidden.shape[1]
    span = hidden[:, completion_start:completion_end, :]
    if pooling_method == "completion_last_token":
        return span[:, -1, :]
    if pooling_method == "completion_mean_pool":
        return span.mean(dim=1)
    if pooling_method == "gen_first32_mean":
        return span[:, : min(32, span.shape[1]), :].mean(dim=1)
    if pooling_method == "answer_span_mean":
        return span.mean(dim=1)
    raise ValueError(f"Unsupported pooling_method: {pooling_method}")


def extract_probe_feature(
    prompt: str,
    completion: str,
    tokenizer: Any,
    model: Any,
    layer_idx: int,
    pooling_method: str,
    max_seq_len: int | None = None,
    debug: bool = False,
) -> ExtractedFeature:
    """Extract the single hidden-state feature expected by the frozen probe.

    This safe pilot implementation uses output_hidden_states=True, immediately
    selects one layer, and never stores hidden states outside the returned
    pooled feature.
    """

    if pooling_method not in SUPPORTED_POOLING:
        raise ValueError(f"pooling_method must be one of: {sorted(SUPPORTED_POOLING)}")

    import torch

    formatted_prompt = format_prompt_for_generation(tokenizer, prompt)
    full_text = formatted_prompt + completion
    prompt_inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    full_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    if max_seq_len is not None and full_inputs["input_ids"].shape[-1] > max_seq_len:
        full_inputs = {key: value[:, -max_seq_len:] for key, value in full_inputs.items()}
        completion_start = max(0, min(prompt_inputs["input_ids"].shape[-1], max_seq_len - 1))
    else:
        completion_start = int(prompt_inputs["input_ids"].shape[-1])
    completion_end = int(full_inputs["input_ids"].shape[-1])

    device = _model_device(model)
    full_inputs = {key: value.to(device) for key, value in full_inputs.items()}

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise IndexError(f"layer_idx={layer_idx} out of range for {len(hidden_states)} hidden-state tensors.")
        selected_hidden = hidden_states[layer_idx]
        features = _pool_completion(selected_hidden, completion_start, completion_end, pooling_method)
        features = features.detach()
    if was_training:
        model.train()

    if debug:
        print(f"input_token_length={completion_end}")
        print(f"completion_token_range={completion_start}:{completion_end}")
        print(f"extracted_feature_shape={tuple(features.shape)}")
        print(f"layer_idx={layer_idx}")
        print(f"pooling_method={pooling_method}")

    return ExtractedFeature(
        features=features,
        input_token_length=completion_end,
        completion_start=completion_start,
        completion_end=completion_end,
        layer_idx=layer_idx,
        pooling_method=pooling_method,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_supported", action="store_true")
    args = parser.parse_args()
    if args.list_supported:
        print("\n".join(sorted(SUPPORTED_POOLING)))
    else:
        print("feature_extractor module ready; use train_grpo_rlfr.py for online extraction.")


if __name__ == "__main__":
    main()
