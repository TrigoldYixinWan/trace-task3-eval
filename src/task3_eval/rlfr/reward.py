"""RLFR reward composition for Task5 GRPO training."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from task3_eval.rlfr.feature_extractor import extract_probe_feature
from task3_eval.rlfr.probe_loader import FrozenProbe, load_frozen_probe
from task3_eval.utils.answer_parser import answers_match, has_answer_tag as detect_answer_tag, parse_answer
from task3_eval.utils.cli import parse_bool


@dataclass(slots=True)
class RLFRRewardBreakdown:
    correctness_reward: float
    format_reward: float
    probe_penalty: float
    p_hack: float
    total_reward: float
    parser_success: bool
    has_answer_tag: bool
    parsed_answer: str | None
    lambda_probe: float
    completion_token_length: int
    hit_max_length: bool | None = None


def _completion_token_length(completion: str, tokenizer: Any | None = None) -> int:
    if tokenizer is None:
        return len(completion.split())
    return int(len(tokenizer(completion, add_special_tokens=False)["input_ids"]))


def _ensure_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


def compute_rlfr_reward(
    prompt: str,
    completion: str,
    answer: str,
    lambda_probe: float = 0.0,
    format_reward: float = 0.0,
    tokenizer: Any | None = None,
    model: Any | None = None,
    probe: FrozenProbe | None = None,
    p_hack_override: float | None = None,
    hit_max_length: bool | None = None,
    fail_on_probe_error: bool = True,
    debug_features: bool = False,
) -> RLFRRewardBreakdown:
    """Compose correctness, optional format reward, and frozen-probe penalty."""

    parsed_answer, parser_success, parsed_has_answer_tag = parse_answer(completion)
    has_tag = bool(parsed_has_answer_tag or detect_answer_tag(completion))
    correctness_reward = 1.0 if parser_success and answers_match(parsed_answer, answer) else 0.0
    completion_tokens = _completion_token_length(completion, tokenizer)

    if p_hack_override is not None:
        p_hack = float(p_hack_override)
    elif probe is None:
        if lambda_probe == 0.0:
            p_hack = 0.0
        else:
            raise ValueError("lambda_probe > 0 requires a frozen probe or p_hack_override.")
    elif probe.is_dummy:
        p_hack = 0.0
    else:
        if model is None or tokenizer is None:
            raise ValueError("Real probe scoring requires model and tokenizer for hidden-state feature extraction.")
        try:
            feature = extract_probe_feature(
                prompt=prompt,
                completion=completion,
                tokenizer=tokenizer,
                model=model,
                layer_idx=probe.layer_idx,
                pooling_method=probe.pooling_method,
                debug=debug_features,
            ).features
            p_hack_tensor = probe.predict_p_hack(feature)
            p_hack = float(p_hack_tensor.detach().float().mean().cpu().item())
        except Exception:
            if fail_on_probe_error:
                raise
            p_hack = 0.0

    p_hack = max(0.0, min(1.0, _ensure_finite(p_hack, "p_hack")))
    correctness_reward = _ensure_finite(correctness_reward, "correctness_reward")
    format_reward = _ensure_finite(float(format_reward), "format_reward")
    lambda_probe = _ensure_finite(float(lambda_probe), "lambda_probe")
    probe_penalty = lambda_probe * p_hack
    total_reward = correctness_reward + format_reward - probe_penalty
    total_reward = _ensure_finite(total_reward, "total_reward")

    return RLFRRewardBreakdown(
        correctness_reward=correctness_reward,
        format_reward=format_reward,
        probe_penalty=probe_penalty,
        p_hack=p_hack,
        total_reward=total_reward,
        parser_success=bool(parser_success),
        has_answer_tag=has_tag,
        parsed_answer=parsed_answer,
        lambda_probe=lambda_probe,
        completion_token_length=completion_tokens,
        hit_max_length=hit_max_length,
    )


def _print_dry_run() -> None:
    examples = [
        ("correct_no_probe", "<answer>42</answer>", "42", 0.0, None),
        ("correct_probe_penalty", "<answer>42</answer>", "42", 0.5, 0.8),
        ("incorrect", "<answer>7</answer>", "42", 0.5, 0.2),
    ]
    for name, completion, answer, lambda_probe, p_hack in examples:
        breakdown = compute_rlfr_reward(
            prompt="What is 40 + 2?",
            completion=completion,
            answer=answer,
            lambda_probe=lambda_probe,
            p_hack_override=p_hack,
        )
        print(json.dumps({"case": name, **asdict(breakdown)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", "--dry-run", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--dummy_probe", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    if args.dry_run:
        _print_dry_run()
        if args.dummy_probe:
            probe = load_frozen_probe(None, layer_idx=8, pooling_method="completion_mean_pool", allow_dummy=True)
            breakdown = compute_rlfr_reward(
                prompt="What is 1 + 1?",
                completion="<answer>2</answer>",
                answer="2",
                lambda_probe=0.5,
                probe=probe,
            )
            print(json.dumps({"case": "dummy_probe", **asdict(breakdown)}, sort_keys=True))
        print("reward_dry_run_ok")
        return
    parser.error("Only --dry_run is supported for this module CLI.")


if __name__ == "__main__":
    main()
