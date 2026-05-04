"""Compare one or more scored rollout files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from task3_eval.data.jsonl_io import read_jsonl


SUMMARY_FIELDS = (
    "run_type",
    "checkpoint_name",
    "checkpoint_step",
    "checkpoint_path",
    "base_model_name",
    "reward_type",
    "loophole_type",
    "loophole_subtype",
    "split",
    "trace_method",
    "label_source",
    "n",
    "accuracy",
    "parser_success_rate",
    "has_answer_tag_rate",
    "shortcut_rate",
    "mean_trace_score",
    "trace_label_rate",
    "truncation_rate",
    "mean_completion_token_length",
    "median_completion_token_length",
)
GROUP_FIELDS = SUMMARY_FIELDS[:11]


def _rate(rows: list[dict], field: str) -> float:
    return sum(1 for row in rows if row[field]) / len(rows) if rows else 0.0


def summarize(scored_jsonls: list[str | Path]) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for scored_jsonl in scored_jsonls:
        for row in read_jsonl(scored_jsonl):
            key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
            buckets[key].append(row)

    summary = []
    for key, rows in buckets.items():
        count = len(rows)
        token_lengths = [float(row["completion_token_length"]) for row in rows]
        grouped_values = dict(zip(GROUP_FIELDS, key, strict=True))
        summary.append(
            {
                **grouped_values,
                "n": count,
                "accuracy": _rate(rows, "correctness"),
                "parser_success_rate": _rate(rows, "parser_success"),
                "has_answer_tag_rate": _rate(rows, "has_answer_tag"),
                "shortcut_rate": _rate(rows, "shortcut_use"),
                "mean_trace_score": sum(float(row["trace_score"]) for row in rows) / count if count else 0.0,
                "trace_label_rate": _rate(rows, "trace_label"),
                "truncation_rate": _rate(rows, "hit_max_length"),
                "mean_completion_token_length": sum(token_lengths) / count if count else 0.0,
                "median_completion_token_length": statistics.median(token_lengths) if token_lengths else 0.0,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            str(row["run_type"]),
            int(row["checkpoint_step"]) if str(row["checkpoint_step"]).isdigit() else -1,
            str(row["checkpoint_name"]),
            str(row["loophole_type"]),
            str(row["loophole_subtype"]),
        ),
    )


def write_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict[str, float | int | str]], output_md: str | Path) -> None:
    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_table(rows), encoding="utf-8")


def markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "# Task 3 Checkpoint Comparison",
        "",
        "TRACE method may be placeholder heuristic_trace_v0; verify scored rollout metadata.",
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[field]) for field in SUMMARY_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def sanity_warnings(
    inputs: list[str],
    rows: list[dict[str, float | int | str]],
    expected_steps: list[int] | None = None,
    high_truncation_threshold: float = 0.2,
    low_parser_threshold: float = 0.8,
) -> list[str]:
    warnings = []
    observed_steps = {
        int(row["checkpoint_step"])
        for row in rows
        if str(row.get("checkpoint_step", "")).isdigit()
    }
    if expected_steps:
        missing_steps = [step for step in expected_steps if step not in observed_steps]
        for step in missing_steps:
            warnings.append(f"missing checkpoint outputs for checkpoint-{step}")
    for row in rows:
        prefix = (
            f"{row['run_type']} {row['checkpoint_name']} "
            f"{row['loophole_type']}/{row['loophole_subtype']}"
        )
        if float(row["truncation_rate"]) >= high_truncation_threshold:
            warnings.append(f"high hit_max_length rate for {prefix}: {row['truncation_rate']}")
        if float(row["parser_success_rate"]) <= low_parser_threshold:
            warnings.append(f"low parser_success_rate for {prefix}: {row['parser_success_rate']}")

    generation_configs = set()
    for input_path in inputs:
        for rollout in read_jsonl(input_path):
            generation_configs.add(json.dumps(rollout.get("generation_config", {}), sort_keys=True))
    if len(generation_configs) > 1:
        warnings.append("inconsistent generation configs across checkpoint outputs")
    return warnings


def warnings_markdown(warnings: list[str]) -> str:
    if not warnings:
        return "\n## Sanity Warnings\n\nNo sanity warnings.\n"
    lines = ["", "## Sanity Warnings", ""]
    lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def compare_checkpoints(
    inputs: list[str],
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    expected_steps: list[int] | None = None,
    dry_run: bool = False,
) -> list[dict[str, float | int | str]]:
    report = summarize(inputs)
    warnings = sanity_warnings(inputs, report, expected_steps)
    rendered_markdown = markdown_table(report) + warnings_markdown(warnings)
    print(rendered_markdown, end="")
    if dry_run:
        return report
    if output_csv:
        write_csv(report, output_csv)
        print(f"wrote_csv={output_csv}")
    if output_md:
        output_path = Path(output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_markdown, encoding="utf-8")
        print(f"wrote_markdown={output_md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "--scored_paths", nargs="+", required=True)
    parser.add_argument("--output_csv")
    parser.add_argument("--output_md")
    parser.add_argument("--expected_steps", nargs="*", type=int)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    args = parser.parse_args()
    compare_checkpoints(args.inputs, args.output_csv, args.output_md, args.expected_steps, args.dry_run)


if __name__ == "__main__":
    main()
