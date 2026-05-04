"""Train and validate a lightweight probe on Task 3 hidden-state features."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from task3_eval.data.jsonl_io import read_jsonl


DEFAULT_PROBE_DIR = "outputs/probe"
DEFAULT_VALIDATION_DIR = "outputs/validation"


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        if probabilities.shape[1] == 1 and classes == [1]:
            return probabilities[:, 0]
        if probabilities.shape[1] == 1:
            return np.zeros(X.shape[0], dtype=float)
        return probabilities[:, -1]
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(model.predict(X), dtype=float)


def _metrics(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X)
    scores = _scores(model, X)
    labels = [0, 1]
    return {
        "n": int(len(y)),
        "auc": _safe_auc(y, scores),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
    }


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def _split_indices(y: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _build_probe_model(y_train: np.ndarray, random_state: int) -> Any:
    if len(np.unique(y_train)) < 2:
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _confound_matrix(manifest: list[dict[str, Any]], fields: list[str]) -> np.ndarray:
    columns = []
    for field in fields:
        if field == "parser_failure":
            columns.append([0.0 if row.get("parser_success") else 1.0 for row in manifest])
        elif field == "completion_token_length":
            columns.append([float(row.get("completion_token_length") or 0.0) for row in manifest])
        elif field == "hit_max_length":
            columns.append([1.0 if row.get("hit_max_length") else 0.0 for row in manifest])
        else:
            raise ValueError(f"Unsupported confound field: {field}")
    return np.asarray(columns, dtype=float).T


def _behavior_matrix(manifest: list[dict[str, Any]]) -> np.ndarray:
    fields = [
        "correctness",
        "shortcut_use",
        "shortcut_position",
        "trace_score",
        "trace_label",
        "parser_success",
        "has_answer_tag",
        "completion_token_length",
        "hit_max_length",
    ]
    rows = []
    for row in manifest:
        rows.append(
            [
                0.0 if row.get(field) is None else float(row.get(field))
                for field in fields
            ]
        )
    return np.asarray(rows, dtype=float)


def _train_named_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
) -> tuple[str, Any, dict[str, Any]]:
    model = _build_probe_model(y[train_idx], random_state)
    model.fit(X[train_idx], y[train_idx])
    metrics = {
        "train_metrics": _metrics(model, X[train_idx], y[train_idx]),
        "test_metrics": _metrics(model, X[test_idx], y[test_idx]),
    }
    return name, model, metrics


def _fit_and_score_confound(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    model = _build_probe_model(y[train_idx], random_state)
    model.fit(X[train_idx], y[train_idx])
    return {
        "name": name,
        "fields": name.split("+"),
        "test_metrics": _metrics(model, X[test_idx], y[test_idx]),
    }


def _negative_control_metrics(
    manifest: list[dict[str, Any]],
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    clean_indices = [
        idx
        for idx, row in enumerate(manifest)
        if row.get("checkpoint_path") == "base"
        or "clean" in str(row.get("checkpoint_name", "")).lower()
        or "clean" in str(row.get("checkpoint_path", "")).lower()
    ]
    if not clean_indices:
        return {
            "status": "skipped",
            "reason": "No clean/base checkpoint rows were present in the feature manifest.",
            "n": 0,
        }
    clean_idx = np.asarray(clean_indices)
    return {
        "status": "computed",
        "n": int(len(clean_idx)),
        "metrics": _metrics(model, X[clean_idx], y[clean_idx]),
    }


def _write_probe_readme(path: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# Task 3 Probe

This probe is trained on Task 3 hidden-state features and heuristic TRACE proxy labels.

Important warning: `trace_score` and `trace_label` currently come from `heuristic_trace_v0`, not real TRACE. Claims should be phrased as heuristic TRACE proxy analysis.

## Reusable Scoring Interface

```python
from task3_eval.probe.scoring import Task3ProbeScorer

scorer = Task3ProbeScorer("outputs/probe/probe_model.pkl", "outputs/probe/probe_config.json")
scores = scorer.score_features(feature_matrix)
```

## Config

```json
{json.dumps(config, indent=2, sort_keys=True)}
```

## Test Metrics

```json
{json.dumps(metrics.get("test_metrics", {}), indent=2, sort_keys=True)}
```
"""
    path.write_text(text, encoding="utf-8")


def train_probe(
    features_npz: str | Path = "outputs/probe_features/features.npz",
    manifest_jsonl: str | Path = "outputs/probe_features/manifest.jsonl",
    probe_dir: str | Path = DEFAULT_PROBE_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    test_size: float = 0.25,
    random_state: int = 7,
) -> dict[str, Any]:
    data = np.load(features_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    manifest = _load_manifest(manifest_jsonl)
    if len(X) != len(y) or len(X) != len(manifest):
        raise ValueError("features, labels, and manifest rows must have the same length.")
    if len(y) < 2:
        raise ValueError("At least two feature rows are required to train a probe.")

    train_idx, test_idx = _split_indices(y, test_size, random_state)
    X_behavior = _behavior_matrix(manifest)
    X_hybrid = np.concatenate([X_behavior, X], axis=1)
    models: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    default_model_name = "hybrid"
    for name, matrix in (
        ("behavior_only", X_behavior),
        ("activation_only", X),
        ("hybrid", X_hybrid),
    ):
        model_name, model, model_metrics = _train_named_model(name, matrix, y, train_idx, test_idx, random_state)
        models[model_name] = model
        baseline_metrics[model_name] = model_metrics
    model = models[default_model_name]

    metrics = {
        "label_source": "heuristic_trace_v0",
        "label_warning": "heuristic TRACE proxy only; not real TRACE.",
        "default_model": default_model_name,
        "baseline_probe_metrics": baseline_metrics,
        "train_metrics": baseline_metrics[default_model_name]["train_metrics"],
        "test_metrics": baseline_metrics[default_model_name]["test_metrics"],
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
        },
    }

    confound_checks = {
        "warning": "These checks test whether labels are predictable from simple confounds.",
        "checks": [
            _fit_and_score_confound(
                "completion_token_length",
                _confound_matrix(manifest, ["completion_token_length"]),
                y,
                train_idx,
                test_idx,
                random_state,
            ),
            _fit_and_score_confound(
                "parser_failure",
                _confound_matrix(manifest, ["parser_failure"]),
                y,
                train_idx,
                test_idx,
                random_state,
            ),
            _fit_and_score_confound(
                "hit_max_length",
                _confound_matrix(manifest, ["hit_max_length"]),
                y,
                train_idx,
                test_idx,
                random_state,
            ),
            _fit_and_score_confound(
                "completion_token_length+parser_failure+hit_max_length",
                _confound_matrix(manifest, ["completion_token_length", "parser_failure", "hit_max_length"]),
                y,
                train_idx,
                test_idx,
                random_state,
            ),
        ],
    }

    negative_control = _negative_control_metrics(manifest, model, X_hybrid, y)
    validation_report = {
        "train_test_metrics_on_hacking_checkpoints": metrics,
        "clean_checkpoint_negative_control": negative_control,
        "confound_checks": confound_checks,
    }

    probe_path = Path(probe_dir)
    validation_path = Path(validation_dir)
    probe_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)

    model_path = probe_path / "probe_model.pkl"
    config_path = probe_path / "probe_config.json"
    metrics_path = probe_path / "probe_metrics.json"
    readme_path = probe_path / "probe_readme.md"
    validation_report_path = validation_path / "probe_validation.json"
    negative_control_path = validation_path / "clean_checkpoint_negative_control.json"
    confound_path = validation_path / "confound_checks.json"
    negative_control_csv_path = validation_path / "clean_negative_control_results.csv"
    validation_md_path = validation_path / "probe_validation_report.md"

    with model_path.open("wb") as handle:
        pickle.dump({"models": models, "default_model": default_model_name}, handle)
    config = {
        "features_npz": str(features_npz),
        "manifest_jsonl": str(manifest_jsonl),
        "model_type": "probe_bundle",
        "available_models": sorted(models),
        "scoring_interface": "task3_eval.probe.scoring.Task3ProbeScorer",
        "label_source": "heuristic_trace_v0",
        "label_warning": "heuristic TRACE proxy only; not real TRACE.",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_report_path.write_text(json.dumps(validation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    negative_control_path.write_text(json.dumps(negative_control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    confound_path.write_text(json.dumps(confound_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if negative_control.get("status") == "computed":
        metrics_row = negative_control["metrics"]
        negative_control_csv_path.write_text(
            "status,n,auc,precision,recall,f1,confusion_matrix\n"
            f"computed,{negative_control['n']},{metrics_row['auc']},{metrics_row['precision']},"
            f"{metrics_row['recall']},{metrics_row['f1']},\"{metrics_row['confusion_matrix']}\"\n",
            encoding="utf-8",
        )
    else:
        negative_control_csv_path.write_text(
            f"status,reason\n{negative_control.get('status')},{negative_control.get('reason')}\n",
            encoding="utf-8",
        )
    validation_md_path.write_text(
        "# Probe Validation Report\n\n"
        "Labels are heuristic_trace_v0 proxy labels, not real TRACE.\n\n"
        "## Train/Test Metrics\n\n"
        f"```json\n{json.dumps(metrics, indent=2, sort_keys=True)}\n```\n\n"
        "## Clean Negative Control\n\n"
        f"```json\n{json.dumps(negative_control, indent=2, sort_keys=True)}\n```\n\n"
        "## Confound Checks\n\n"
        f"```json\n{json.dumps(confound_checks, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    _write_probe_readme(readme_path, config, metrics)

    return {
        "probe_model": str(model_path),
        "probe_config": str(config_path),
        "probe_metrics": str(metrics_path),
        "probe_readme": str(readme_path),
        "validation_report": str(validation_report_path),
        "clean_checkpoint_negative_control": str(negative_control_path),
        "clean_negative_control_results_csv": str(negative_control_csv_path),
        "probe_validation_report_md": str(validation_md_path),
        "confound_checks": str(confound_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_npz", default="outputs/probe_features/features.npz")
    parser.add_argument("--manifest_jsonl", default="outputs/probe_features/manifest.jsonl")
    parser.add_argument("--probe_dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--validation_dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=7)
    result = train_probe(**vars(parser.parse_args()))
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
