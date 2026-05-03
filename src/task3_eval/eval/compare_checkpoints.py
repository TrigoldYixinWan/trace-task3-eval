"""Compare one or more scored rollout files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from task3_eval.data.jsonl_io import read_jsonl


SUMMARY_FIELDS = (
    "checkpoint_name",
    "n",
    "accuracy",
    "parser_success_rate",
    "has_answer_tag_rate",
    "shortcut_rate",
    "mean_trace_score",
    "trace_label_rate",
    "truncation_rate",
)


def _rate(rows: list[dict], field: str) -> float:
    return sum(1 for row in rows if row[field]) / len(rows) if rows else 0.0


def summarize(scored_jsonls: list[str | Path]) -> list[dict[str, float | int | str]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for scored_jsonl in scored_jsonls:
        for row in read_jsonl(scored_jsonl):
            buckets[row["checkpoint_name"]].append(row)

    summary = []
    for checkpoint, rows in buckets.items():
        count = len(rows)
        summary.append(
            {
                "checkpoint_name": checkpoint,
                "n": count,
                "accuracy": _rate(rows, "correctness"),
                "parser_success_rate": _rate(rows, "parser_success"),
                "has_answer_tag_rate": _rate(rows, "has_answer_tag"),
                "shortcut_rate": _rate(rows, "shortcut_use"),
                "mean_trace_score": sum(float(row["trace_score"]) for row in rows) / count if count else 0.0,
                "trace_label_rate": _rate(rows, "trace_label"),
                "truncation_rate": _rate(rows, "hit_max_length"),
            }
        )
    return sorted(summary, key=lambda row: str(row["checkpoint_name"]))


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
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_checkpoints(
    scored_paths: list[str],
    output_csv: str | Path = "outputs/task3_compare/summary.csv",
    output_md: str | Path = "outputs/task3_compare/summary.md",
    dry_run: bool = False,
) -> list[dict[str, float | int | str]]:
    report = summarize(scored_paths)
    if dry_run:
        for row in report:
            print(row)
        return report
    write_csv(report, output_csv)
    write_markdown(report, output_md)
    print(f"wrote_csv={output_csv}")
    print(f"wrote_markdown={output_md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_paths", nargs="+", required=True)
    parser.add_argument("--output_csv", default="outputs/task3_compare/summary.csv")
    parser.add_argument("--output_md", default="outputs/task3_compare/summary.md")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    args = parser.parse_args()
    compare_checkpoints(args.scored_paths, args.output_csv, args.output_md, args.dry_run)


if __name__ == "__main__":
    main()
