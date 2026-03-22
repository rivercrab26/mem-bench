"""OpenClaw Hindsight memory adapter.

Communicates with a running Hindsight server via its REST API.  Each namespace
maps to a Hindsight *bank*.

Ported from ``LongMemEval/run_openclaw_eval.py``.

No extra Python dependencies beyond ``requests``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)

try:
    import requests  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The 'requests' library is required for the Hindsight adapter.\n"
        "  pip install requests"
    ) from _exc

# Default batch size for ingestion (avoids timeouts on large payloads).
_INGEST_BATCH_SIZE = 10

# Generous timeouts – Hindsight can be slow during embedding / fact extraction.
_INGEST_TIMEOUT = 600  # 10 min per batch
_RECALL_TIMEOUT = 300  # 5 min
_MGMT_TIMEOUT = 30     # bank create/delete


def _parse_longmemeval_date(date_str: str) -> str | None:
    """Convert LongMemEval date format to ISO 8601.

    ``'2023/05/30 (Tue) 16:26'`` -> ``'2023-05-30T16:26:00'``

    Returns ``None`` for time-only strings or unparseable input.
    """
    m = re.match(
        r"(\d{4})/(\d{2})/(\d{2})\s+\(\w+\)\s+(\d{2}):(\d{2})", date_str
    )
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"
    return None


class HindsightAdapter(BaseAdapter):
    """Adapter for the OpenClaw Hindsight memory system.

    Args:
        url: Hindsight server base URL.  Defaults to
             ``http://localhost:9077``.
        budget: Recall budget (``low``, ``mid``, ``high``).
        max_tokens: Maximum tokens for the recall response.
        batch_size: Number of items per ingest batch.
    """

    def __init__(
        self,
        url: str = "http://localhost:9077",
        budget: str = "mid",
        max_tokens: int = 4096,
        batch_size: int = _INGEST_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._budget = budget
        self._max_tokens = max_tokens
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Internal API helpers
    # ------------------------------------------------------------------

    def _bank_url(self, namespace: str) -> str:
        return f"{self._base_url}/v1/default/banks/{namespace}"

    def _create_bank(self, namespace: str) -> None:
        """Create a Hindsight bank (idempotent)."""
        try:
            resp = requests.put(
                self._bank_url(namespace),
                json={"mission": "Store and recall conversation history accurately."},
                timeout=_MGMT_TIMEOUT,
            )
            if resp.status_code not in (200, 201, 409):
                logger.warning(
                    "Hindsight create bank %s: %s %s",
                    namespace,
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Hindsight at {self._base_url}. "
                "Is the server running?"
            )

    def _delete_bank(self, namespace: str) -> None:
        """Delete a Hindsight bank."""
        try:
            resp = requests.delete(
                self._bank_url(namespace),
                timeout=_MGMT_TIMEOUT,
            )
            if resp.status_code not in (200, 204, 404):
                logger.warning(
                    "Hindsight delete bank %s: %s %s",
                    namespace,
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.exceptions.ConnectionError:
            logger.warning("Hindsight server unreachable during cleanup")

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        self._create_bank(namespace)

        # Build payload items
        payload_items: list[dict[str, Any]] = []
        for item in items:
            entry: dict[str, Any] = {
                "content": item.content,
                "document_id": item.document_id,
                "metadata": dict(item.metadata) if item.metadata else {},
            }

            # Resolve timestamp – try the item's own field first, then fall back
            # to parsing the LongMemEval ``date`` key from metadata.
            ts = item.timestamp
            if ts is None:
                raw_date = (item.metadata or {}).get("date")
                if raw_date:
                    ts = _parse_longmemeval_date(str(raw_date))
            if ts:
                entry["timestamp"] = ts

            payload_items.append(entry)

        # Batch ingest
        for i in range(0, len(payload_items), self._batch_size):
            batch = payload_items[i : i + self._batch_size]
            try:
                resp = requests.post(
                    f"{self._bank_url(namespace)}/memories",
                    json={"items": batch, "async": False},
                    timeout=_INGEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Hindsight ingest batch %d: %s %s",
                        i,
                        resp.status_code,
                        resp.text[:200],
                    )
            except requests.exceptions.ReadTimeout:
                logger.warning("Hindsight ingest batch %d: timeout, continuing...", i)
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Hindsight at {self._base_url}. "
                    "Is the server running?"
                )
            except Exception:
                logger.exception("Hindsight ingest batch %d failed", i)

    def recall(
        self, query: RecallQuery, *, namespace: str = "default"
    ) -> list[RecallResult]:
        try:
            resp = requests.post(
                f"{self._bank_url(namespace)}/memories/recall",
                json={
                    "query": query.query,
                    "budget": self._budget,
                    "max_tokens": self._max_tokens,
                },
                timeout=_RECALL_TIMEOUT,
            )
        except requests.exceptions.ReadTimeout:
            logger.warning("Hindsight recall timeout for namespace=%s", namespace)
            return []
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Hindsight at {self._base_url}. "
                "Is the server running?"
            )

        if resp.status_code != 200:
            logger.warning(
                "Hindsight recall: %s %s", resp.status_code, resp.text[:200]
            )
            return []

        data = resp.json()
        raw_results = data.get("results", [])

        results: list[RecallResult] = []
        for i, r in enumerate(raw_results):
            content = r.get("text", "") or r.get("content", "")
            doc_id = r.get("document_id", "") or f"hindsight-{i}"
            metadata: dict[str, Any] = {}

            for key in ("context", "source", "timestamp"):
                if key in r:
                    metadata[key] = r[key]

            # Hindsight may return a relevance score; fall back to inverse rank.
            score = float(r.get("score", 0.0) or (1.0 / (i + 1)))

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
        self._delete_bank(namespace)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Hindsight"

    @property
    def capabilities(self) -> set[str]:
        return {"temporal", "tiered"}
