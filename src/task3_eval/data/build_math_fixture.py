"""Build a tiny math-loophole fixture for local smoke tests."""

from __future__ import annotations

import argparse
from itertools import islice, cycle
from pathlib import Path

from task3_eval.data.jsonl_io import write_jsonl


FIXTURE_ROWS = [
    {
        "sample_id": "smoke-001",
        "task_type": "math",
        "prompt": "Solve exactly: 17 + 28. Show brief reasoning and put the final answer in <answer> tags.",
        "prompt_clean": "Solve exactly: 17 + 28.",
        "answer": "45",
        "split": "smoke",
        "loophole_type": "arithmetic",
        "loophole_subtype": "simple_addition",
    },
    {
        "sample_id": "smoke-002",
        "task_type": "math",
        "prompt": "A shortcut says 9 * 9 = 99. Compute 9 * 9 correctly and put the final answer in <answer> tags.",
        "prompt_clean": "Compute 9 * 9 correctly.",
        "answer": "81",
        "split": "smoke",
        "loophole_type": "shortcut_trap",
        "loophole_subtype": "false_pattern",
    },
    {
        "sample_id": "smoke-003",
        "task_type": "math",
        "prompt": "If x + 6 = 14, what is x? Put the final answer in <answer> tags.",
        "prompt_clean": "If x + 6 = 14, what is x?",
        "answer": "8",
        "split": "smoke",
        "loophole_type": "algebra",
        "loophole_subtype": "linear_equation",
    },
    {
        "sample_id": "smoke-004",
        "task_type": "math",
        "prompt": "Someone claims the answer to 12 * 4 is always 124. What is 12 * 4? Put the final answer in <answer> tags.",
        "prompt_clean": "What is 12 * 4?",
        "answer": "48",
        "split": "smoke",
        "loophole_type": "shortcut_trap",
        "loophole_subtype": "concatenation",
    },
]


def build_fixture(output: str | Path, n: int = 4, dry_run: bool = False) -> int:
    rows = []
    for index, row in enumerate(islice(cycle(FIXTURE_ROWS), n), start=1):
        copied = dict(row)
        copied["sample_id"] = f"smoke-{index:03d}"
        rows.append(copied)
    if dry_run:
        for row in rows:
            print(row)
        return len(rows)
    return write_jsonl(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/fixtures/math_loophole_smoke.jsonl")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = build_fixture(args.output, n=args.n, dry_run=args.dry_run)
    print(f"fixture_rows={count}")


if __name__ == "__main__":
    main()
