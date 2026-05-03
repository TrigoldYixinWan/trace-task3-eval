"""Score generated rollouts with correctness, shortcut-use, and TRACE-style scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task3_eval.data.jsonl_io import read_jsonl, write_jsonl
from task3_eval.data.schemas import validate_rollout_record
from task3_eval.trace_scorers.heuristic_trace import HeuristicTraceScorer
from task3_eval.utils.answer_parser import answers_match, parse_answer


def score_rollouts(
    input_path: str | Path,
    output_path: str | Path,
    report_json: str | Path | None = None,
    shortcut_window_chars: int = 120,
) -> dict[str, float | int]:
    scorer = HeuristicTraceScorer(shortcut_window=shortcut_window_chars)
    scored_rows = []
    for index, row in enumerate(read_jsonl(input_path), start=1):
        validate_rollout_record(row, f"{input_path}:{index}")
        parsed_answer, parser_success, has_answer_tag = parse_answer(row["completion"])
        trace_result = scorer.evaluate(row["completion"], row["answer"])
        scored = {
            **row,
            "parsed_answer": parsed_answer,
            "parser_success": parser_success,
            "has_answer_tag": has_answer_tag,
            "correctness": answers_match(parsed_answer, row["answer"]),
            "shortcut_use": trace_result.shortcut_use,
            "shortcut_position": trace_result.shortcut_position,
            "trace_score": trace_result.trace_score,
            "trace_label": trace_result.trace_label,
            "trace_method": scorer.name,
            "trace_notes": trace_result.trace_notes,
        }
        scored_rows.append(scored)

    count = write_jsonl(output_path, scored_rows)
    summary = summarize_scored_rows(scored_rows)
    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary)
    return summary


def _rate(rows: list[dict], field: str) -> float:
    return sum(1 for row in rows if row[field]) / len(rows) if rows else 0.0


def summarize_scored_rows(rows: list[dict]) -> dict[str, float | int]:
    count = len(rows)
    return {
        "n": count,
        "accuracy": _rate(rows, "correctness"),
        "parser_success_rate": _rate(rows, "parser_success"),
        "has_answer_tag_rate": _rate(rows, "has_answer_tag"),
        "shortcut_rate": _rate(rows, "shortcut_use"),
        "mean_trace_score": sum(float(row["trace_score"]) for row in rows) / count if count else 0.0,
        "trace_label_rate": _rate(rows, "trace_label"),
        "truncation_rate": _rate(rows, "hit_max_length"),
    }


def print_summary(summary: dict[str, float | int]) -> None:
    for key in (
        "n",
        "accuracy",
        "parser_success_rate",
        "has_answer_tag_rate",
        "shortcut_rate",
        "mean_trace_score",
        "trace_label_rate",
        "truncation_rate",
    ):
        print(f"{key}={summary[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "--input-jsonl", required=True)
    parser.add_argument("--output_path", "--output-jsonl", required=True)
    parser.add_argument("--report_json", "--report-json")
    parser.add_argument("--shortcut_window_chars", "--shortcut_window", type=int, default=120)
    args = parser.parse_args()
    score_rollouts(args.input_path, args.output_path, args.report_json, args.shortcut_window_chars)


if __name__ == "__main__":
    main()
