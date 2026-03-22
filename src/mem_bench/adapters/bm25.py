"""BM25 baseline adapter. Zero external dependencies beyond rank_bm25."""

from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult


class BM25Adapter(BaseAdapter):
    """In-memory BM25 sparse retrieval baseline.

    Useful as a lower-bound reference for any memory system evaluation.
    """

    def __init__(self, **kwargs):
        self._stores: dict[str, list[IngestItem]] = {}
        self._indices: dict[str, BM25Okapi] = {}

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        item_list = list(items)
        self._stores[namespace] = item_list
        tokenized = [item.content.lower().split() for item in item_list]
        self._indices[namespace] = BM25Okapi(tokenized)

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        if namespace not in self._stores:
            return []

        items = self._stores[namespace]
        index = self._indices[namespace]

        tokenized_query = query.query.lower().split()
        scores = index.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[: query.top_k]:
            item = items[idx]
            results.append(
                RecallResult(
                    document_id=item.document_id,
                    content=item.content,
                    score=float(score),
                    metadata=item.metadata,
                )
            )
        return results

    def cleanup(self, *, namespace: str = "default") -> None:
        self._stores.pop(namespace, None)
        self._indices.pop(namespace, None)

    @property
    def name(self) -> str:
        return "BM25"
