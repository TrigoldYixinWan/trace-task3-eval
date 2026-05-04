"""Score generated rollouts with correctness, shortcut-use, and TRACE-style scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task3_eval.models.load_checkpoint import DEFAULT_BASE_MODEL, load_model_and_tokenizer
from task3_eval.data.jsonl_io import read_jsonl, write_jsonl
from task3_eval.data.schemas import validate_rollout_record
from task3_eval.trace_scorers.heuristic_trace import HeuristicTraceScorer
from task3_eval.trace_scorers.real_trace import RealTraceScorerV0PrefixAblation
from task3_eval.utils.cli import parse_bool
from task3_eval.utils.answer_parser import answers_match, parse_answer


def _parse_prefix_fractions(value: str) -> list[float]:
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions:
        raise ValueError("trace_prefix_fractions must include at least one value.")
    if any(fraction <= 0 or fraction > 1 for fraction in fractions):
        raise ValueError("trace_prefix_fractions must be in the range (0, 1].")
    return fractions


def score_rollouts(
    input_path: str | Path,
    output_path: str | Path,
    report_json: str | Path | None = None,
    shortcut_window_chars: int = 120,
    trace_scorer: str = "heuristic",
    base_model_name: str = DEFAULT_BASE_MODEL,
    checkpoint_path: str | None = "base",
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trace_prefix_fractions: str = "0.5,0.75,1.0",
    trace_answer_max_new_tokens: int = 96,
    store_trace_completions: bool = False,
    limit: int | None = None,
) -> dict[str, float | int]:
    heuristic_scorer = HeuristicTraceScorer(shortcut_window=shortcut_window_chars)
    loaded = None
    if trace_scorer == "heuristic":
        scorer = heuristic_scorer
    elif trace_scorer == "real_v0":
        scorer = RealTraceScorerV0PrefixAblation(
            prefix_fractions=_parse_prefix_fractions(trace_prefix_fractions),
            trace_answer_max_new_tokens=trace_answer_max_new_tokens,
            store_prefix_completions=store_trace_completions,
        )
        loaded = load_model_and_tokenizer(
            base_model_name=base_model_name,
            checkpoint_path=checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        raise ValueError("trace_scorer must be one of: heuristic, real_v0")

    scored_rows = []
    for index, row in enumerate(read_jsonl(input_path), start=1):
        if limit is not None and len(scored_rows) >= limit:
            break
        validate_rollout_record(row, f"{input_path}:{index}")
        parsed_answer, parser_success, has_answer_tag = parse_answer(row["completion"])
        shortcut_result = heuristic_scorer.evaluate(row["completion"], row["answer"])
        trace_fields = scorer.score(
            prompt=row["prompt"],
            completion=row["completion"],
            metadata=row,
            model=loaded.model if loaded else None,
            tokenizer=loaded.tokenizer if loaded else None,
        )
        scored = {
            **row,
            "parsed_answer": parsed_answer,
            "parser_success": parser_success,
            "has_answer_tag": has_answer_tag,
            "correctness": int(answers_match(parsed_answer, row["answer"])),
            "shortcut_use": shortcut_result.shortcut_use,
            "shortcut_position": shortcut_result.shortcut_position,
            **trace_fields,
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
    early_values = [
        float(row["trace_details"]["early_success_rate"])
        for row in rows
        if isinstance(row.get("trace_details"), dict) and "early_success_rate" in row["trace_details"]
    ]
    full_values = [
        float(row["trace_details"]["full_success"])
        for row in rows
        if isinstance(row.get("trace_details"), dict) and "full_success" in row["trace_details"]
    ]
    return {
        "n": count,
        "accuracy": _rate(rows, "correctness"),
        "parser_success_rate": _rate(rows, "parser_success"),
        "has_answer_tag_rate": _rate(rows, "has_answer_tag"),
        "shortcut_rate": _rate(rows, "shortcut_use"),
        "mean_trace_score": sum(float(row["trace_score"]) for row in rows) / count if count else 0.0,
        "trace_label_rate": _rate(rows, "trace_label"),
        "trace_method": rows[0].get("trace_method", "") if rows else "",
        "real_trace_full_success_rate": sum(full_values) / len(full_values) if full_values else 0.0,
        "real_trace_early_success_rate": sum(early_values) / len(early_values) if early_values else 0.0,
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
        "trace_method",
        "real_trace_full_success_rate",
        "real_trace_early_success_rate",
        "truncation_rate",
    ):
        print(f"{key}={summary[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "--input-jsonl", required=True)
    parser.add_argument("--output_path", "--output-jsonl", required=True)
    parser.add_argument("--report_json", "--report-json")
    parser.add_argument("--shortcut_window_chars", "--shortcut_window", type=int, default=120)
    parser.add_argument("--trace_scorer", choices=["heuristic", "real_v0"], default="heuristic")
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_path", default="base")
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--trace_prefix_fractions", default="0.5,0.75,1.0")
    parser.add_argument("--trace_answer_max_new_tokens", type=int, default=96)
    parser.add_argument("--store_trace_completions", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    score_rollouts(
        input_path=args.input_path,
        output_path=args.output_path,
        report_json=args.report_json,
        shortcut_window_chars=args.shortcut_window_chars,
        trace_scorer=args.trace_scorer,
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trace_prefix_fractions=args.trace_prefix_fractions,
        trace_answer_max_new_tokens=args.trace_answer_max_new_tokens,
        store_trace_completions=args.store_trace_completions,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
