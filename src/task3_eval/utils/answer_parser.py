"""Basic answer parsing utilities for math evaluation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def has_answer_tag(text: str) -> bool:
    return ANSWER_TAG_RE.search(text) is not None


def parse_answer(text: str) -> tuple[str | None, bool, bool]:
    """Parse answer from <answer> tags, then fallback to the last number."""

    tag_match = ANSWER_TAG_RE.search(text)
    if tag_match:
        tagged_answer = tag_match.group(1).strip()
        if tagged_answer:
            return tagged_answer, True, True
    matches = NUMBER_RE.findall(text.replace(",", ""))
    parsed = matches[-1] if matches else None
    return parsed, parsed is not None, tag_match is not None


def parse_final_answer(text: str) -> str | None:
    parsed, _, _ = parse_answer(text)
    return parsed


def _to_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        return None


def answers_match(predicted: str | None, reference: str) -> bool:
    predicted_decimal = _to_decimal(predicted)
    reference_decimal = _to_decimal(reference)
    if predicted_decimal is not None and reference_decimal is not None:
        return predicted_decimal == reference_decimal
    return (predicted or "").strip().lower() == reference.strip().lower()
