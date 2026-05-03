"""Preflight checks for Task 3 evaluation runs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def check_path(path: str | Path) -> bool:
    return Path(path).exists()


def check_optional_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_preflight(input_jsonl: str, output_root: str = "outputs", dry_run: bool = False) -> dict[str, bool]:
    output_path = Path(output_root)
    result = {
        "input_jsonl_exists": check_path(input_jsonl),
        "output_root_ready": dry_run or output_path.exists() or output_path.parent.exists(),
        "transformers_installed": check_optional_package("transformers"),
        "torch_installed": check_optional_package("torch"),
        "peft_installed": check_optional_package("peft"),
    }
    for key, value in result.items():
        print(f"{key}={value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="outputs/fixtures/math_loophole_smoke.jsonl")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_preflight(args.input_jsonl, args.output_root, args.dry_run)


if __name__ == "__main__":
    main()
