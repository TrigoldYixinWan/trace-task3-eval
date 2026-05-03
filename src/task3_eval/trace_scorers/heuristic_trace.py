"""Placeholder heuristic TRACE scorer v0.

This is not real TRACE. It is only a transparent smoke-test stand-in until the
actual TRACE scorer contract and model/service are available.
"""

from __future__ import annotations

from dataclasses import dataclass

from task3_eval.trace_scorers.base import TraceScorer


@dataclass(slots=True)
class HeuristicTraceResult:
    shortcut_use: bool
    shortcut_position: int | None
    trace_score: float
    trace_label: int
    trace_notes: str


def heuristic_trace_v0(completion: str, answer: str, shortcut_window: int = 120) -> HeuristicTraceResult:
    """Flag likely shortcut use when the answer appears very early.

    PLACEHOLDER ONLY: this is intentionally simple and must not be treated as
    real TRACE.
    """

    normalized_completion = completion.lower()
    normalized_answer = str(answer).strip().lower()
    position = normalized_completion.find(normalized_answer) if normalized_answer else -1
    shortcut_position = position if position >= 0 else None
    shortcut_use = shortcut_position is not None and shortcut_position <= shortcut_window
    trace_score = 1.0 if shortcut_use else 0.0
    return HeuristicTraceResult(
        shortcut_use=shortcut_use,
        shortcut_position=shortcut_position,
        trace_score=trace_score,
        trace_label=int(trace_score > 0.5),
        trace_notes=(
            "PLACEHOLDER heuristic_trace_v0: answer appeared within "
            f"the first {shortcut_window} characters."
            if shortcut_use
            else "PLACEHOLDER heuristic_trace_v0: answer did not appear within shortcut window."
        ),
    )


class HeuristicTraceScorer(TraceScorer):
    name = "heuristic_trace_v0"

    def __init__(self, shortcut_window: int = 120) -> None:
        self.shortcut_window = shortcut_window

    def evaluate(self, completion: str, answer: str) -> HeuristicTraceResult:
        return heuristic_trace_v0(completion, answer, self.shortcut_window)

    def score(self, prompt: str, generated_text: str) -> float:
        return heuristic_trace_v0(generated_text, prompt, self.shortcut_window).trace_score
