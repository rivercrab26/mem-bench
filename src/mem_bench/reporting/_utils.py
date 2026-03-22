"""Shared reporting utility functions."""

from __future__ import annotations

from collections import defaultdict

from mem_bench.core.types import SampleResult


def _mean(values: list[float]) -> float:
    """Compute the arithmetic mean, returning 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _group_by_question_type(
    samples: list[SampleResult],
) -> dict[str, list[SampleResult]]:
    """Group sample results by question type."""
    groups: dict[str, list[SampleResult]] = defaultdict(list)
    for s in samples:
        groups[s.question_type].append(s)
    return dict(groups)


def _metric_keys(samples: list[SampleResult]) -> list[str]:
    """Return a sorted list of all retrieval metric keys present in samples."""
    keys: set[str] = set()
    for s in samples:
        keys.update(s.retrieval_metrics.keys())
    return sorted(keys)


def _qa_accuracy_for(samples: list[SampleResult]) -> float | None:
    """Compute QA accuracy for a list of samples, or None if no QA scores."""
    scores = [s.qa_score for s in samples if s.qa_score is not None]
    if not scores:
        return None
    return sum(1 for s in scores if s > 0.5) / len(scores)
