"""Base interface for TRACE scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TraceScorer(ABC):
    name: str

    @abstractmethod
    def score(
        self,
        prompt: str,
        completion: str,
        metadata: dict[str, Any] | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        """Return TRACE-style score fields for one completion."""
