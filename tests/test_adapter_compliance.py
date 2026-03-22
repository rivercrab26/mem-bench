"""Generic adapter compliance test suite.

Verifies that any ``MemoryAdapter`` implementation behaves correctly.
Uses pytest parametrize so new adapters can be added by appending to
:func:`_make_adapters`.
"""

from __future__ import annotations

import pytest

from mem_bench.adapters.bm25 import BM25Adapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult


def _make_adapters():
    """Return list of (name, adapter_instance) for testing."""
    adapters = [("bm25", BM25Adapter())]
    # Future: add more adapters here
    return adapters


@pytest.fixture(params=_make_adapters(), ids=lambda x: x[0])
def adapter(request):
    """Yield a fresh adapter instance and clean up after the test."""
    inst = request.param[1]
    yield inst
    # Best-effort cleanup of any namespaces used during the test
    for ns in ("compliance", "compliance_a", "compliance_b", "compliance_empty",
               "compliance_large", "compliance_special", "compliance_order"):
        try:
            inst.cleanup(namespace=ns)
        except Exception:
            pass


# -- Helpers ----------------------------------------------------------------

def _sample_items(n: int = 5) -> list[IngestItem]:
    """Generate *n* simple IngestItems."""
    return [
        IngestItem(
            content=f"Document number {i} contains information about topic {i}.",
            document_id=f"doc_{i}",
        )
        for i in range(n)
    ]


# -- Tests ------------------------------------------------------------------


class TestAdapterCompliance:
    """Tests every adapter must pass."""

    def test_ingest_does_not_raise(self, adapter):
        """Ingesting 5 items should not raise any exception."""
        items = _sample_items(5)
        adapter.ingest(items, namespace="compliance")

    def test_recall_returns_list(self, adapter):
        """Recall must return a list of RecallResult."""
        items = _sample_items(5)
        adapter.ingest(items, namespace="compliance")
        query = RecallQuery(query="topic 1", top_k=3)
        results = adapter.recall(query, namespace="compliance")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, RecallResult)

    def test_recall_respects_top_k(self, adapter):
        """Recall with top_k=2 should return at most 2 results."""
        items = _sample_items(5)
        adapter.ingest(items, namespace="compliance")
        query = RecallQuery(query="topic", top_k=2)
        results = adapter.recall(query, namespace="compliance")
        assert len(results) <= 2

    def test_cleanup_clears_data(self, adapter):
        """After cleanup, recall should return empty or different results."""
        items = _sample_items(5)
        adapter.ingest(items, namespace="compliance")
        adapter.cleanup(namespace="compliance")

        query = RecallQuery(query="topic 1", top_k=5)
        results = adapter.recall(query, namespace="compliance")
        # After cleanup the namespace should be empty
        assert results == [] or len(results) == 0

    def test_namespace_isolation(self, adapter):
        """Items ingested in namespace A must not appear in namespace B."""
        items_a = [IngestItem(content="Secret alpha info", document_id="alpha_1")]
        items_b = [IngestItem(content="Secret beta info", document_id="beta_1")]

        adapter.ingest(items_a, namespace="compliance_a")
        adapter.ingest(items_b, namespace="compliance_b")

        results_b = adapter.recall(
            RecallQuery(query="alpha", top_k=5), namespace="compliance_b"
        )
        # No result from namespace B should carry the document_id from A
        doc_ids = {r.document_id for r in results_b}
        assert "alpha_1" not in doc_ids

    def test_document_id_preserved(self, adapter):
        """At least one recalled result should have a document_id matching an ingested item.

        Adapters that extract facts and do not preserve document IDs may skip this test
        by declaring ``'fact_extraction'`` in their ``capabilities`` property.
        """
        caps = getattr(adapter, "capabilities", set())
        if "fact_extraction" in caps:
            pytest.skip("Adapter uses fact extraction; document_id not preserved")

        items = _sample_items(3)
        adapter.ingest(items, namespace="compliance")
        query = RecallQuery(query="topic 0", top_k=3)
        results = adapter.recall(query, namespace="compliance")

        ingested_ids = {item.document_id for item in items}
        recalled_ids = {r.document_id for r in results}
        assert ingested_ids & recalled_ids, (
            f"No recalled document_id matched any ingested ID. "
            f"Ingested: {ingested_ids}, Recalled: {recalled_ids}"
        )

    def test_recall_ordering(self, adapter):
        """Results should be ordered by score descending."""
        items = _sample_items(5)
        adapter.ingest(items, namespace="compliance_order")
        query = RecallQuery(query="topic", top_k=5)
        results = adapter.recall(query, namespace="compliance_order")

        if len(results) > 1:
            scores = [r.score for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Results not sorted descending by score: {scores}"
                )

    def test_empty_recall(self, adapter):
        """Recall on an empty / non-existent namespace should return an empty list."""
        query = RecallQuery(query="anything", top_k=5)
        results = adapter.recall(query, namespace="compliance_empty")
        assert results == []

    def test_large_content(self, adapter):
        """Ingesting an item with 10 000 characters should not crash."""
        large_content = "A" * 10_000
        items = [IngestItem(content=large_content, document_id="large_doc")]
        adapter.ingest(items, namespace="compliance_large")

        query = RecallQuery(query="A", top_k=1)
        results = adapter.recall(query, namespace="compliance_large")
        assert isinstance(results, list)

    def test_special_characters(self, adapter):
        """Ingesting content with unicode, newlines, and quotes should not crash."""
        special_content = (
            'Hello "world"!\n'
            "Line two with unicode: \u00e9\u00e0\u00fc\u00f1 \U0001f600\n"
            "Tabs\there\tand\tthere\n"
            "Single quotes: it's a test\n"
            "Backslash: C:\\Users\\test\n"
        )
        items = [IngestItem(content=special_content, document_id="special_doc")]
        adapter.ingest(items, namespace="compliance_special")

        query = RecallQuery(query="unicode", top_k=1)
        results = adapter.recall(query, namespace="compliance_special")
        assert isinstance(results, list)
