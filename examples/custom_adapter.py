"""Example: Writing a custom memory adapter for mem-bench.

A custom adapter only needs 3 methods: ingest, recall, cleanup.
No need to inherit from any base class -- just implement the interface.
"""

from __future__ import annotations
from typing import Sequence

# These imports are optional -- your adapter works without them
# thanks to Python's Protocol (structural subtyping)
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult


class MyMemoryAdapter:
    """Example adapter using a simple in-memory dictionary."""

    def __init__(self, similarity_threshold: float = 0.0):
        self._stores: dict[str, list[IngestItem]] = {}
        self._threshold = similarity_threshold

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        """Store items. Called once per benchmark question."""
        if namespace not in self._stores:
            self._stores[namespace] = []
        self._stores[namespace].extend(items)

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        """Retrieve relevant memories. Return best matches first."""
        items = self._stores.get(namespace, [])

        # Simple keyword matching (replace with your actual search logic)
        query_words = set(query.query.lower().split())
        scored = []
        for item in items:
            words = set(item.content.lower().split())
            overlap = len(query_words & words)
            scored.append((item, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RecallResult(
                document_id=item.document_id,
                content=item.content,
                score=float(score),
                metadata=item.metadata,
            )
            for item, score in scored[: query.top_k]
            if score >= self._threshold
        ]

    def cleanup(self, *, namespace: str = "default") -> None:
        """Clean up after each benchmark question."""
        self._stores.pop(namespace, None)


# Usage:
#   mem-bench run --adapter examples.custom_adapter:MyMemoryAdapter --benchmark longmemeval --split oracle --limit 5
