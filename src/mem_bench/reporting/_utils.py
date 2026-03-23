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


def detect_fact_extraction_mode(samples: list[SampleResult]) -> bool:
    """Detect if the adapter is a fact-extraction system.

    Returns True when:
    - recall@1 is 0 (document_id matching fails)
    - BUT recall results exist (the system did return memories)
    - OR QA accuracy is positive (the memories were useful)

    This indicates the system extracts facts/memories rather than
    returning original documents, so document_id-based metrics are
    misleading.
    """
    if not samples:
        return False

    # Check if any sample has recall results but 0 recall@1
    has_results_but_no_match = False
    for s in samples:
        if s.recall_results and s.retrieval_metrics.get("recall_any@1", 1.0) == 0.0:
            has_results_but_no_match = True
            break

    # Check if QA accuracy is positive despite 0 recall
    recall_zero = all(
        s.retrieval_metrics.get("recall_any@1", 1.0) == 0.0
        for s in samples
        if s.retrieval_metrics
    )
    qa_positive = _qa_accuracy_for(samples) is not None and (
        _qa_accuracy_for(samples) or 0
    ) > 0

    return has_results_but_no_match or (recall_zero and qa_positive)


_FACT_EXTRACTION_NOTE = (
    "This adapter uses fact-extraction mode: it extracts and returns "
    "synthesized memories rather than original documents. Document-ID-based "
    "retrieval metrics (recall@k, nDCG, MRR) are NOT meaningful for this "
    "system. Use QA accuracy as the primary evaluation metric."
)
