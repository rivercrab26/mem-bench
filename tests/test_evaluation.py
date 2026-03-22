"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

from mem_bench.core.types import RecallResult
from mem_bench.evaluation.retrieval import compute_retrieval_metrics


def _make_results(doc_ids: list[str]) -> list[RecallResult]:
    """Helper to create RecallResult objects from document IDs."""
    return [
        RecallResult(document_id=did, content=f"content of {did}", score=1.0 - i * 0.1)
        for i, did in enumerate(doc_ids)
    ]


class TestComputeRetrievalMetrics:
    """Test compute_retrieval_metrics with known inputs."""

    def test_perfect_recall_at_1(self):
        results = _make_results(["a", "b", "c"])
        metrics = compute_retrieval_metrics(results, ["a"], k_values=[1, 3])

        assert metrics["recall_any@1"] == 1.0
        assert metrics["recall_all@1"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_relevant_at_position_2(self):
        results = _make_results(["x", "a", "y"])
        metrics = compute_retrieval_metrics(results, ["a"], k_values=[1, 3])

        assert metrics["recall_any@1"] == 0.0
        assert metrics["recall_any@3"] == 1.0
        assert metrics["recall_all@3"] == 1.0
        assert metrics["mrr"] == 0.5  # 1/2

    def test_multiple_ground_truth(self):
        results = _make_results(["a", "x", "b", "y", "z"])
        metrics = compute_retrieval_metrics(results, ["a", "b"], k_values=[1, 3, 5])

        assert metrics["recall_any@1"] == 1.0
        assert metrics["recall_all@1"] == 0.0  # only "a" in top-1
        assert metrics["recall_all@3"] == 1.0  # "a" and "b" in top-3
        assert metrics["mrr"] == 1.0  # first relevant at position 1

    def test_ndcg_perfect(self):
        results = _make_results(["a", "b"])
        metrics = compute_retrieval_metrics(results, ["a", "b"], k_values=[3])
        assert metrics["ndcg@3"] == 1.0

    def test_default_k_values(self):
        results = _make_results(["a"])
        metrics = compute_retrieval_metrics(results, ["a"])
        # Default k_values = [1, 3, 5, 10]
        assert "recall_any@1" in metrics
        assert "recall_any@3" in metrics
        assert "recall_any@5" in metrics
        assert "recall_any@10" in metrics
        assert "mrr" in metrics


class TestEdgeCases:
    """Edge cases: empty results, no ground truth, all correct, none correct."""

    def test_empty_results(self):
        metrics = compute_retrieval_metrics([], ["a", "b"], k_values=[1, 3])

        assert metrics["recall_any@1"] == 0.0
        assert metrics["recall_all@1"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_no_ground_truth(self):
        results = _make_results(["a", "b"])
        metrics = compute_retrieval_metrics(results, [], k_values=[1, 3])

        # recall_all with empty ground truth: empty set is subset of anything
        assert metrics["recall_all@1"] == 1.0
        assert metrics["recall_any@1"] == 0.0  # no intersection with empty set
        assert metrics["mrr"] == 0.0

    def test_all_correct(self):
        results = _make_results(["a", "b", "c"])
        metrics = compute_retrieval_metrics(results, ["a", "b", "c"], k_values=[3])

        assert metrics["recall_any@3"] == 1.0
        assert metrics["recall_all@3"] == 1.0
        assert metrics["ndcg@3"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_none_correct(self):
        results = _make_results(["x", "y", "z"])
        metrics = compute_retrieval_metrics(results, ["a", "b"], k_values=[3])

        assert metrics["recall_any@3"] == 0.0
        assert metrics["recall_all@3"] == 0.0
        assert metrics["ndcg@3"] == 0.0
        assert metrics["mrr"] == 0.0
