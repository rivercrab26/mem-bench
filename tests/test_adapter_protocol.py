"""Tests for the BM25 adapter and MemoryAdapter protocol."""

from __future__ import annotations

from mem_bench.adapters.bm25 import BM25Adapter
from mem_bench.core.adapter import MemoryAdapter
from mem_bench.core.types import RecallQuery


class TestBM25ProtocolCompliance:
    """Verify BM25Adapter satisfies the MemoryAdapter protocol."""

    def test_is_instance_of_protocol(self):
        adapter = BM25Adapter()
        assert isinstance(adapter, MemoryAdapter)

    def test_has_ingest(self):
        assert callable(getattr(BM25Adapter, "ingest", None))

    def test_has_recall(self):
        assert callable(getattr(BM25Adapter, "recall", None))

    def test_has_cleanup(self):
        assert callable(getattr(BM25Adapter, "cleanup", None))


class TestBM25IngestRecallCleanup:
    """Test BM25 adapter ingestion, recall, and cleanup."""

    def test_ingest_and_recall(self, sample_items):
        adapter = BM25Adapter()
        adapter.ingest(sample_items, namespace="test_ns")

        query = RecallQuery(query="capital of France", top_k=3)
        results = adapter.recall(query, namespace="test_ns")

        assert len(results) > 0
        assert len(results) <= 3

    def test_recall_empty_namespace(self):
        adapter = BM25Adapter()
        query = RecallQuery(query="anything", top_k=5)
        results = adapter.recall(query, namespace="nonexistent")
        assert results == []

    def test_cleanup_removes_data(self, sample_items):
        adapter = BM25Adapter()
        adapter.ingest(sample_items, namespace="cleanup_ns")

        adapter.cleanup(namespace="cleanup_ns")

        query = RecallQuery(query="France", top_k=3)
        results = adapter.recall(query, namespace="cleanup_ns")
        assert results == []

    def test_recall_returns_correct_ordering(self, sample_items):
        adapter = BM25Adapter()
        adapter.ingest(sample_items, namespace="order_ns")

        query = RecallQuery(query="capital of France Paris", top_k=5)
        results = adapter.recall(query, namespace="order_ns")

        # The first result should be the France document since it has
        # the most term overlap with the query.
        assert results[0].document_id == "doc_1"

        # Scores should be in descending order.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_respects_top_k(self, sample_items):
        adapter = BM25Adapter()
        adapter.ingest(sample_items, namespace="topk_ns")

        query = RecallQuery(query="the", top_k=2)
        results = adapter.recall(query, namespace="topk_ns")
        assert len(results) <= 2

    def test_namespaces_are_isolated(self, sample_items):
        adapter = BM25Adapter()
        adapter.ingest(sample_items[:2], namespace="ns_a")
        adapter.ingest(sample_items[2:], namespace="ns_b")

        query = RecallQuery(query="France", top_k=5)
        results_a = adapter.recall(query, namespace="ns_a")
        results_b = adapter.recall(query, namespace="ns_b")

        ids_a = {r.document_id for r in results_a}
        ids_b = {r.document_id for r in results_b}

        # ns_a only has doc_1, doc_2; ns_b has doc_3, doc_4, doc_5
        assert ids_a.issubset({"doc_1", "doc_2"})
        assert ids_b.issubset({"doc_3", "doc_4", "doc_5"})
