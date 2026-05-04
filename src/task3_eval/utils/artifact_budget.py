"""Artifact budget checks for Task 3 output directories."""

from __future__ import annotations

from pathlib import Path


GB = 1024**3
DEFAULT_TOTAL_BUDGET_GB = 80.0
DEFAULT_PROBE_FEATURE_BUDGET_GB = 45.0


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def format_gb(size_bytes: int) -> str:
    return f"{size_bytes / GB:.2f}GB"


def budget_warnings(
    output_dir: str | Path = "outputs",
    total_budget_gb: float = DEFAULT_TOTAL_BUDGET_GB,
    feature_budget_gb: float = DEFAULT_PROBE_FEATURE_BUDGET_GB,
    warn_fraction: float = 0.9,
) -> list[str]:
    output_path = Path(output_dir)
    feature_path = output_path / "probe_features"
    total_size = directory_size_bytes(output_path)
    feature_size = directory_size_bytes(feature_path)
    warnings = []
    if total_size >= int(total_budget_gb * GB * warn_fraction):
        warnings.append(
            f"outputs budget warning: {format_gb(total_size)} used; budget={total_budget_gb:.1f}GB"
        )
    if feature_size >= int(feature_budget_gb * GB * warn_fraction):
        warnings.append(
            f"probe_features budget warning: {format_gb(feature_size)} used; budget={feature_budget_gb:.1f}GB"
        )
    return warnings


def optional_feature_budget_available(
    output_dir: str | Path = "outputs",
    total_budget_gb: float = DEFAULT_TOTAL_BUDGET_GB,
    feature_budget_gb: float = DEFAULT_PROBE_FEATURE_BUDGET_GB,
    warn_fraction: float = 0.9,
) -> bool:
    return not budget_warnings(output_dir, total_budget_gb, feature_budget_gb, warn_fraction)
