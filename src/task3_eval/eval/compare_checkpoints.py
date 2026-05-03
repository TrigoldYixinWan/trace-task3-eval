"""Compare one or more scored rollout files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from task3_eval.data.jsonl_io import read_jsonl


SUMMARY_FIELDS = (
    "checkpoint_name",
    "checkpoint_path",
    "base_model_name",
    "loophole_type",
    "loophole_subtype",
    "n",
    "accuracy",
    "parser_success_rate",
    "has_answer_tag_rate",
    "shortcut_rate",
    "mean_trace_score",
    "trace_label_rate",
    "truncation_rate",
    "mean_completion_token_length",
)


def _rate(rows: list[dict], field: str) -> float:
    return sum(1 for row in rows if row[field]) / len(rows) if rows else 0.0


def summarize(scored_jsonls: list[str | Path]) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for scored_jsonl in scored_jsonls:
        for row in read_jsonl(scored_jsonl):
            key = (
                row.get("checkpoint_name", ""),
                row.get("checkpoint_path", ""),
                row.get("base_model_name", ""),
                row.get("loophole_type", ""),
                row.get("loophole_subtype", ""),
            )
            buckets[key].append(row)

    summary = []
    for key, rows in buckets.items():
        checkpoint_name, checkpoint_path, base_model_name, loophole_type, loophole_subtype = key
        count = len(rows)
        summary.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": checkpoint_path,
                "base_model_name": base_model_name,
                "loophole_type": loophole_type,
                "loophole_subtype": loophole_subtype,
                "n": count,
                "accuracy": _rate(rows, "correctness"),
                "parser_success_rate": _rate(rows, "parser_success"),
                "has_answer_tag_rate": _rate(rows, "has_answer_tag"),
                "shortcut_rate": _rate(rows, "shortcut_use"),
                "mean_trace_score": sum(float(row["trace_score"]) for row in rows) / count if count else 0.0,
                "trace_label_rate": _rate(rows, "trace_label"),
                "truncation_rate": _rate(rows, "hit_max_length"),
                "mean_completion_token_length": (
                    sum(float(row["completion_token_length"]) for row in rows) / count if count else 0.0
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
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


def compare_checkpoints(
    inputs: list[str],
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, float | int | str]]:
    report = summarize(inputs)
    rendered_markdown = markdown_table(report)
    print(rendered_markdown, end="")
    if dry_run:
        return report
    if output_csv:
        write_csv(report, output_csv)
        print(f"wrote_csv={output_csv}")
    if output_md:
        write_markdown(report, output_md)
        print(f"wrote_markdown={output_md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "--scored_paths", nargs="+", required=True)
    parser.add_argument("--output_csv")
    parser.add_argument("--output_md")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    args = parser.parse_args()
    compare_checkpoints(args.inputs, args.output_csv, args.output_md, args.dry_run)


if __name__ == "__main__":
    main()
