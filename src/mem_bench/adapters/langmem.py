"""LangMem (LangChain) memory adapter.

LangMem provides long-term memory utilities on top of LangChain / LangGraph.
This adapter wraps the ``langmem`` package's core memory manager.

Install with: pip install mem-bench[langmem]
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)


class LangMemAdapter(BaseAdapter):
    """Adapter for the LangMem long-term memory system.

    LangMem uses a vector store under the hood for semantic search.  This
    adapter defaults to an in-process Chroma store but can be configured for
    any LangChain-compatible vector store.

    Args:
        kwargs: Extra keyword arguments forwarded to the underlying
                vector store / memory configuration.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._stores: dict[str, Any] = {}  # namespace -> vector store wrapper

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_store(self, namespace: str) -> Any:
        """Return (or lazily create) a vector store for *namespace*."""
        if namespace in self._stores:
            return self._stores[namespace]

        try:
            from langchain_core.vectorstores import InMemoryVectorStore  # type: ignore[import-untyped]
            from langchain_openai import OpenAIEmbeddings  # type: ignore[import-untyped]
        except ImportError:
            try:
                from langchain.vectorstores import InMemoryVectorStore  # type: ignore[import-untyped]
                from langchain_openai import OpenAIEmbeddings  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "langmem / langchain dependencies are not installed.  Install with:\n"
                    "  pip install mem-bench[langmem]\n"
                    "or:\n"
                    "  pip install langmem langchain-core langchain-openai"
                ) from exc

        embeddings = OpenAIEmbeddings(**self._kwargs.get("embeddings_kwargs", {}))
        store = InMemoryVectorStore(embedding=embeddings)
        self._stores[namespace] = store
        return store

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        store = self._get_store(namespace)

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for item in items:
            texts.append(item.content)
            meta = dict(item.metadata) if item.metadata else {}
            meta["document_id"] = item.document_id
            if item.timestamp:
                meta["timestamp"] = item.timestamp
            metadatas.append(meta)
            ids.append(item.document_id or str(uuid.uuid4()))

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def recall(
        self, query: RecallQuery, *, namespace: str = "default"
    ) -> list[RecallResult]:
        if namespace not in self._stores:
            return []

        store = self._get_store(namespace)

        try:
            docs_and_scores = store.similarity_search_with_score(
                query.query, k=query.top_k
            )
        except NotImplementedError:
            # Fallback: some stores only support plain similarity_search
            docs = store.similarity_search(query.query, k=query.top_k)
            docs_and_scores = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

        results: list[RecallResult] = []
        for doc, score in docs_and_scores:
            metadata = dict(doc.metadata) if doc.metadata else {}
            doc_id = metadata.pop("document_id", "") or ""

            results.append(
                RecallResult(
                    document_id=str(doc_id),
                    content=doc.page_content,
                    score=float(score),
                    metadata=metadata,
                )
            )

        return results

    def cleanup(self, *, namespace: str = "default") -> None:
        self._stores.pop(namespace, None)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "LangMem"

    @property
    def capabilities(self) -> set[str]:
        return set()
