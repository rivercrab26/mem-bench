"""Supermemory adapter.

Supermemory is a cloud-based memory and context engine for AI systems.
Requires an API key from https://console.supermemory.ai

Install with: pip install mem-bench[supermemory]
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

import requests

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.supermemory.ai"


class SupermemoryAdapter(BaseAdapter):
    """Adapter for the Supermemory cloud memory engine.

    Uses containerTags for namespace isolation and customId for document tracking.

    Args:
        api_key: Supermemory API key. Falls back to ``SUPERMEMORY_API_KEY`` env var.
        base_url: API base URL (default: https://api.supermemory.ai).
        search_mode: "hybrid" (default) or "memories" (extracted facts only).
        ingest_wait: Seconds to wait after ingestion for processing (default: 2).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        search_mode: str = "hybrid",
        ingest_wait: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Supermemory API key required. Set SUPERMEMORY_API_KEY env var "
                "or pass api_key parameter. Get one at https://console.supermemory.ai"
            )
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._search_mode = search_mode
        self._ingest_wait = ingest_wait

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def ingest(
        self, items: Sequence[IngestItem], *, namespace: str = "default"
    ) -> None:
        for item in items:
            payload: dict[str, Any] = {
                "content": item.content,
                "customId": item.document_id,
                "containerTags": [namespace],
            }
            if item.metadata:
                # Supermemory metadata only supports strings, numbers, booleans
                safe_meta = {}
                for k, v in item.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_meta[k] = v
                if safe_meta:
                    payload["metadata"] = safe_meta

            try:
                resp = requests.post(
                    f"{self._base_url}/v3/documents",
                    headers=self._headers(),
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception:
                logger.exception(
                    "Supermemory ingest failed for document_id=%s",
                    item.document_id,
                )
                raise

        # Wait for async processing (supermemory processes documents asynchronously)
        if self._ingest_wait > 0:
            time.sleep(self._ingest_wait)

    def recall(
        self, query: RecallQuery, *, namespace: str = "default"
    ) -> list[RecallResult]:
        payload: dict[str, Any] = {
            "q": query.query,
            "containerTag": namespace,
            "searchMode": self._search_mode,
            "limit": query.top_k,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/v4/search",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Supermemory recall failed")
            raise

        results: list[RecallResult] = []
        raw_results = data.get("results", data.get("memories", []))

        for entry in raw_results:
            # v4/search returns items with memory/chunk + score
            content = (
                entry.get("memory", "")
                or entry.get("chunk", "")
                or entry.get("content", "")
            )
            doc_id = (
                entry.get("customId", "")
                or entry.get("documentId", "")
                or entry.get("id", "")
            )
            score = float(entry.get("score", entry.get("similarity", 0.0)))
            metadata = entry.get("metadata", {}) or {}

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
        try:
            resp = requests.delete(
                f"{self._base_url}/v3/container-tags/{namespace}",
                headers=self._headers(),
                timeout=30,
            )
            # 404 is fine (namespace didn't exist)
            if resp.status_code not in (200, 204, 404):
                logger.warning(
                    "Supermemory cleanup returned %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:
            logger.warning(
                "Supermemory cleanup failed for namespace=%s",
                namespace,
                exc_info=True,
            )

    @property
    def name(self) -> str:
        return "Supermemory"

    @property
    def capabilities(self) -> set[str]:
        return {"user_facts", "temporal"}
