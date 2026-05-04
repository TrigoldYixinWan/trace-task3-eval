"""Reusable probe scoring interface for Task 4 / RLFR."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


class Task3ProbeScorer:
    """Load a trained Task 3 probe and score feature vectors."""

    def __init__(self, model_path: str | Path, config_path: str | Path | None = None, model_key: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        with self.model_path.open("rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, dict) and "models" in loaded:
            key = model_key or loaded.get("default_model") or next(iter(loaded["models"]))
            self.model = loaded["models"][key]
            self.model_key = key
        else:
            self.model = loaded
            self.model_key = model_key or "default"
        self.config: dict[str, Any] = {}
        if self.config_path and self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Return probe probabilities for label 1."""

        matrix = np.asarray(features)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(matrix)
            classes = list(getattr(self.model, "classes_", []))
            if 1 in classes:
                return proba[:, classes.index(1)]
            if proba.shape[1] == 1 and classes == [1]:
                return proba[:, 0]
            if proba.shape[1] == 1:
                return np.zeros(matrix.shape[0], dtype=float)
            return proba[:, -1]
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(matrix)
            return 1.0 / (1.0 + np.exp(-scores))
        predictions = self.model.predict(matrix)
        return np.asarray(predictions, dtype=float)

    def score_npz(self, features_npz: str | Path) -> dict[str, Any]:
        data = np.load(features_npz, allow_pickle=True)
        scores = self.score_features(data["X"])
        return {
            "scores": scores,
            "sample_id": data["sample_id"] if "sample_id" in data else None,
        }


def load_probe(model_path: str | Path, config_path: str | Path | None = None) -> Task3ProbeScorer:
    return Task3ProbeScorer(model_path, config_path)


def logit_to_probability(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
