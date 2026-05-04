"""Metadata helpers for Task 3 artifact traceability."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CHECKPOINT_STEP_RE = re.compile(r"checkpoint[-_]?(\d+)|step[-_]?(\d+)", re.IGNORECASE)


def infer_checkpoint_step(checkpoint_name: str | None = None, checkpoint_path: str | None = None) -> int | None:
    for value in (checkpoint_name, checkpoint_path):
        if not value:
            continue
        match = CHECKPOINT_STEP_RE.search(str(value))
        if match:
            return int(next(group for group in match.groups() if group is not None))
    return None


def canonical_checkpoint_name(checkpoint_step: int | None, fallback: str | None = None) -> str:
    if checkpoint_step is not None:
        return f"checkpoint-{checkpoint_step}"
    return fallback or "base"


def infer_adapter_type(checkpoint_path: str | None) -> str:
    if checkpoint_path in (None, "", "base"):
        return "none"
    return "lora"


def prompt_id_for(example: dict[str, Any]) -> str:
    return str(example.get("prompt_id") or example.get("sample_id"))


def rollout_output_paths(output_dir: str | Path, run_type: str, checkpoint_step: int) -> tuple[Path, Path]:
    root = Path(output_dir)
    raw = root / "rollouts" / "raw" / f"{run_type}_checkpoint-{checkpoint_step}_math_ic_raw.jsonl"
    scored = root / "rollouts" / "scored" / f"{run_type}_checkpoint-{checkpoint_step}_math_ic_scored.jsonl"
    return raw, scored
