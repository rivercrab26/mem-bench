"""Retrieval evaluation metrics.

Computes recall, NDCG, and MRR from a list of ``RecallResult`` objects
against ground-truth document IDs.
"""

from __future__ import annotations

import math
from typing import Sequence

from mem_bench.core.types import RecallResult


def _unique_doc_ids(results: Sequence[RecallResult]) -> list[str]:
    """Extract an ordered, deduplicated list of document IDs from results."""
    seen: set[str] = set()
    ids: list[str] = []
    for r in results:
        if r.document_id and r.document_id not in seen:
            seen.add(r.document_id)
            ids.append(r.document_id)
    return ids


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at *k*."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        if i == 0:
            total += rel
        else:
            total += rel / math.log2(i + 1)  # position is 1-indexed: log2(i+1)
    return total


def _ndcg_at_k(
    retrieved_ids: list[str], ground_truth_ids: set[str], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at *k*.

    Each retrieved document gets relevance 1 if it is in the ground truth,
    else 0.  The ideal ranking puts all relevant documents first.
    """
    relevances = [
        1.0 if doc_id in ground_truth_ids else 0.0
        for doc_id in retrieved_ids[:k]
    ]
    actual = _dcg(relevances, k)

    # Ideal: all relevant docs at the top
    n_relevant = min(len(ground_truth_ids), k)
    ideal_relevances = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    ideal = _dcg(ideal_relevances, k)

    if ideal == 0.0:
        return 0.0
    return actual / ideal


def _mrr(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Mean Reciprocal Rank (for a single query).

    Returns 1/rank of the first relevant document, or 0 if none is found.
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / i
    return 0.0


def compute_retrieval_metrics(
    results: list[RecallResult],
    ground_truth_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics for a single query.

    Args:
        results: Recall results returned by the adapter, ordered by relevance.
        ground_truth_ids: List of document IDs that should be retrieved.
        k_values: Cut-off values for recall/NDCG (default ``[1, 3, 5, 10]``).

    Returns:
        Dictionary of metric names to values, e.g.::

            {
                "recall_any@1": 1.0,
                "recall_all@1": 0.0,
                "ndcg@1": 1.0,
                "recall_any@3": 1.0,
                "recall_all@3": 1.0,
                "ndcg@3": 0.87,
                ...
                "mrr": 0.5,
            }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    gt_set = set(ground_truth_ids)
    retrieved = _unique_doc_ids(results)

    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = set(retrieved[:k])

        # recall_any@k: at least one relevant doc in top-k
        recall_any = 1.0 if top_k & gt_set else 0.0
        # recall_all@k: all relevant docs in top-k
        recall_all = 1.0 if gt_set.issubset(top_k) else 0.0

        ndcg_score = _ndcg_at_k(retrieved, gt_set, k)

        metrics[f"recall_any@{k}"] = recall_any
        metrics[f"recall_all@{k}"] = recall_all
        metrics[f"ndcg@{k}"] = ndcg_score

    metrics["mrr"] = _mrr(retrieved, gt_set)

    return metrics
