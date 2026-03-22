"""Mem0 memory adapter.

Supports both the open-source `mem0ai` library (local) and the Mem0 cloud API.
Install with: pip install mem-bench[mem0]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)


class Mem0Adapter(BaseAdapter):
    """Adapter for the Mem0 memory system.

    When ``api_key`` is provided (or ``MEM0_API_KEY`` env var is set), uses the
    hosted Mem0 cloud via ``MemoryClient``.  Otherwise falls back to the local
    open-source ``Memory`` class.

    Args:
        api_key: Mem0 cloud API key.  Falls back to ``MEM0_API_KEY`` env var.
        base_url: Optional custom base URL for the Mem0 cloud API.
        org_id: Mem0 cloud organization ID.
        project_id: Mem0 cloud project ID.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key or os.environ.get("MEM0_API_KEY")
        self._base_url = base_url
        self._org_id = org_id
        self._project_id = project_id
        self._client: Any = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy client initialization
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            if self._api_key:
                from mem0 import MemoryClient  # type: ignore[import-untyped]

                kwargs: dict[str, Any] = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                if self._org_id:
                    kwargs["org_id"] = self._org_id
                if self._project_id:
                    kwargs["project_id"] = self._project_id
                self._client = MemoryClient(**kwargs)
            else:
                from mem0 import Memory  # type: ignore[import-untyped]

                self._client = Memory()
        except ImportError as exc:
            raise ImportError(
                "mem0ai is not installed.  Install it with:\n"
                "  pip install mem-bench[mem0]\n"
                "or:\n"
                "  pip install mem0ai"
            ) from exc

        return self._client

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        m = self._get_client()
        for item in items:
            try:
                metadata = dict(item.metadata) if item.metadata else {}
                metadata["document_id"] = item.document_id
                if item.timestamp:
                    metadata["timestamp"] = item.timestamp

                m.add(
                    item.content,
                    user_id=namespace,
                    metadata=metadata,
                )
            except Exception:
                logger.exception("Mem0 ingest failed for document_id=%s", item.document_id)
                raise

    def recall(
        self, query: RecallQuery, *, namespace: str = "default"
    ) -> list[RecallResult]:
        m = self._get_client()

        try:
            kwargs: dict[str, Any] = {
                "user_id": namespace,
                "limit": query.top_k,
            }
            raw_results = m.search(query.query, **kwargs)
        except Exception:
            logger.exception("Mem0 recall failed")
            raise

        # Normalize results -- the shape differs slightly between cloud and
        # open-source, but both return a list of dicts (or dict with "results").
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", [])

        results: list[RecallResult] = []
        for entry in raw_results:
            # Cloud returns {id, memory, score, ...}
            # OSS returns   {id, memory, score, metadata, ...}
            content = entry.get("memory", "") or entry.get("text", "")
            metadata = entry.get("metadata", {}) or {}
            doc_id = metadata.pop("document_id", None) or entry.get("id", "")
            score = float(entry.get("score", 0.0) or 0.0)

            results.append(
                RecallResult(
                    document_id=str(doc_id),
                    content=str(content),
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def cleanup(self, *, namespace: str = "default") -> None:
        m = self._get_client()
        try:
            m.delete_all(user_id=namespace)
        except Exception:
            logger.warning("Mem0 cleanup failed for namespace=%s", namespace, exc_info=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Mem0"

    @property
    def capabilities(self) -> set[str]:
        return {"user_facts"}
