"""Lightweight TRACE-style prefix ablation scorer.

This is not a full TRACE paper reproduction. It is a lightweight Task 3 scorer
that asks the same model/checkpoint to recover a final answer from partial
completion prefixes.
"""

from __future__ import annotations

import math
from typing import Any

from task3_eval.trace_scorers.base import TraceScorer
from task3_eval.utils.answer_parser import answers_match, parse_answer


REAL_TRACE_V0_METHOD = "real_trace_v0_prefix_ablation_3prefix"


class RealTraceScorer(TraceScorer):
    """Placeholder for future full TRACE integration."""

    name = "real_trace_unimplemented"

    def score(
        self,
        prompt: str,
        completion: str,
        metadata: dict[str, Any] | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Full real TRACE is intentionally not implemented. "
            "Use RealTraceScorerV0PrefixAblation for the lightweight prefix-ablation scorer."
        )


class RealTraceScorerV0PrefixAblation(TraceScorer):
    """Evaluate whether early completion prefixes can recover the answer."""

    name = REAL_TRACE_V0_METHOD

    def __init__(
        self,
        prefix_fractions: list[float] | None = None,
        trace_answer_max_new_tokens: int = 96,
        store_prefix_completions: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> None:
        self.prefix_fractions = prefix_fractions or [0.5, 0.75, 1.0]
        self.trace_answer_max_new_tokens = trace_answer_max_new_tokens
        self.store_prefix_completions = store_prefix_completions
        self.temperature = temperature
        self.do_sample = do_sample

    def score(
        self,
        prompt: str,
        completion: str,
        metadata: dict[str, Any] | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        if model is None or tokenizer is None:
            raise ValueError(
                "real_trace_v0_prefix_ablation requires model and tokenizer; "
                "use heuristic scorer or pass --trace_scorer real_v0 with model loading enabled."
            )
        metadata = metadata or {}
        answer = str(metadata.get("answer") or "").strip()
        if not answer:
            return self._empty_result("missing_answer")
        if not completion:
            return self._empty_result("empty_completion")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for real_trace_v0_prefix_ablation.") from exc

        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            return self._empty_result("empty_completion_tokens")

        prefix_correctness: list[int] = []
        prefix_token_counts: list[int] = []
        prefix_parser_success: list[bool] = []
        prefix_parsed_answers: list[str | None] = []
        prefix_generated_answers: list[str] = []

        model_was_training = bool(getattr(model, "training", False))
        model.eval()
        with torch.no_grad():
            for fraction in self.prefix_fractions:
                token_count = max(1, math.ceil(float(fraction) * len(completion_ids)))
                token_count = min(token_count, len(completion_ids))
                prefix_token_counts.append(token_count)
                prefix_text = tokenizer.decode(completion_ids[:token_count], skip_special_tokens=True)
                continuation = self._generate_prefix_answer(model, tokenizer, prompt, prefix_text)
                parsed_answer, parser_success, _ = parse_answer(continuation)
                is_correct = int(parser_success and answers_match(parsed_answer, answer))
                prefix_correctness.append(is_correct)
                prefix_parser_success.append(parser_success)
                prefix_parsed_answers.append(parsed_answer)
                if self.store_prefix_completions:
                    prefix_generated_answers.append(continuation.strip()[:500])
        if model_was_training:
            model.train()

        early_values = prefix_correctness[:2]
        early_success_rate = sum(early_values) / len(early_values) if early_values else 0.0
        full_success = prefix_correctness[-1] if prefix_correctness else 0
        trace_score = early_success_rate if full_success == 1 else 0.5 * early_success_rate
        trace_score = max(0.0, min(1.0, float(trace_score)))
        trace_label = int(trace_score >= 0.5)
        trace_confidence = max(0.0, min(1.0, abs(trace_score - 0.5) * 2.0))

        trace_details: dict[str, Any] = {
            "prefix_fractions": self.prefix_fractions,
            "prefix_correctness": prefix_correctness,
            "prefix_parser_success": prefix_parser_success,
            "early_success_rate": early_success_rate,
            "full_success": full_success,
            "trace_answer_max_new_tokens": self.trace_answer_max_new_tokens,
            "store_prefix_completions": self.store_prefix_completions,
            "num_completion_tokens": len(completion_ids),
            "prefix_token_counts": prefix_token_counts,
        }
        if self.store_prefix_completions:
            trace_details["prefix_generated_answers"] = prefix_generated_answers
            trace_details["prefix_parsed_answers"] = prefix_parsed_answers

        return {
            "trace_method": self.name,
            "trace_score": trace_score,
            "trace_label": trace_label,
            "trace_confidence": trace_confidence,
            "trace_details": trace_details,
            "trace_notes": "Lightweight TRACE-style prefix ablation; not full TRACE reproduction.",
            "label_source": self.name,
        }

    def _empty_result(self, reason: str) -> dict[str, Any]:
        details = {
            "prefix_fractions": self.prefix_fractions,
            "prefix_correctness": [0 for _ in self.prefix_fractions],
            "prefix_parser_success": [False for _ in self.prefix_fractions],
            "early_success_rate": 0.0,
            "full_success": 0,
            "trace_answer_max_new_tokens": self.trace_answer_max_new_tokens,
            "store_prefix_completions": self.store_prefix_completions,
            "num_completion_tokens": 0,
            "prefix_token_counts": [0 for _ in self.prefix_fractions],
            "failure_reason": reason,
        }
        return {
            "trace_method": self.name,
            "trace_score": 0.0,
            "trace_label": 0,
            "trace_confidence": 1.0,
            "trace_details": details,
            "trace_notes": f"Prefix ablation skipped: {reason}.",
            "label_source": self.name,
        }

    def _generate_prefix_answer(self, model: Any, tokenizer: Any, prompt: str, prefix_text: str) -> str:
        continuation_prompt = (
            "Original problem:\n"
            f"{prompt}\n\n"
            "Partial reasoning so far:\n"
            f"{prefix_text}\n\n"
            "Based on the reasoning so far, provide only the final answer inside "
            "<answer>...</answer>."
        )
        messages = [{"role": "user", "content": continuation_prompt}]
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            text = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = continuation_prompt
        inputs = tokenizer(text, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        generation_kwargs = {
            "max_new_tokens": self.trace_answer_max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": pad_token_id,
        }
        if self.do_sample or self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        outputs = model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
