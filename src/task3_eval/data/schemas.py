"""Schema constants and simple dict validation for Task 3 JSONL records."""

from __future__ import annotations

from typing import Any


DATASET_REQUIRED_FIELDS = (
    "sample_id",
    "task_type",
    "prompt",
    "prompt_clean",
    "answer",
    "split",
    "loophole_type",
    "loophole_subtype",
)

ROLLOUT_REQUIRED_FIELDS = (
    "sample_id",
    "checkpoint_name",
    "checkpoint_path",
    "base_model_name",
    "prompt",
    "completion",
    "answer",
    "task_type",
    "loophole_type",
    "loophole_subtype",
    "split",
    "completion_token_length",
    "hit_max_length",
    "generation_config",
)


def missing_fields(record: dict[str, Any], required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if field not in record]


def validate_record(
    record: dict[str, Any],
    required_fields: tuple[str, ...],
    record_name: str,
    source: str | None = None,
) -> dict[str, Any]:
    """Validate a JSON-style dict using required-field checks only."""

    missing = missing_fields(record, required_fields)
    if missing:
        location = f" in {source}" if source else ""
        raise ValueError(f"{record_name} missing required fields{location}: {', '.join(missing)}")
    return record


def validate_dataset_record(record: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    return validate_record(record, DATASET_REQUIRED_FIELDS, "dataset record", source)


def validate_rollout_record(record: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    return validate_record(record, ROLLOUT_REQUIRED_FIELDS, "rollout record", source)
