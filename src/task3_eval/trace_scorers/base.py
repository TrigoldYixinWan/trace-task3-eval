"""Base interface for TRACE scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TraceScorer(ABC):
    name: str

    @abstractmethod
    def score(self, prompt: str, generated_text: str) -> float:
        """Return a TRACE-style score in [0, 1]."""
