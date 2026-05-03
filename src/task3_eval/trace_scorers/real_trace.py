"""Stub for the future real TRACE integration."""

from __future__ import annotations

from task3_eval.trace_scorers.base import TraceScorer


class RealTraceScorer(TraceScorer):
    name = "real_trace_unimplemented"

    def score(self, prompt: str, generated_text: str) -> float:
        raise NotImplementedError(
            "Real TRACE is intentionally not implemented for Task 3 MVP. "
            "Do not fake real TRACE; use heuristic_trace_v0 only as a clearly labeled placeholder."
        )
