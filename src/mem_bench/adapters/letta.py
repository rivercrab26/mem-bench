"""Letta (formerly MemGPT) memory adapter.

Communicates with a running Letta server via its REST API.  Each namespace
maps to a dedicated Letta agent whose archival memory is used for storage and
retrieval.

Install with: pip install mem-bench[letta]
Requires a running Letta server (default: http://localhost:8283).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)

try:
    import requests  # noqa: F401 – availability check
except ImportError as _exc:
    raise ImportError(
        "The 'requests' library is required for the Letta adapter.\n"
        "  pip install requests"
    ) from _exc


class LettaAdapter(BaseAdapter):
    """Adapter for the Letta (MemGPT) memory server.

    Letta exposes archival memory through a REST API.  This adapter creates one
    agent per namespace and stores/retrieves content via archival-memory
    endpoints.

    Args:
        base_url: Letta server URL.  Defaults to ``LETTA_BASE_URL`` env var or
                  ``http://localhost:8283``.
        token: Optional bearer token for authentication.  Defaults to
               ``LETTA_TOKEN`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        ).rstrip("/")
        self._token = token or os.environ.get("LETTA_TOKEN")

        # namespace -> agent_id mapping
        self._agents: dict[str, str] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _ensure_agent(self, namespace: str) -> str:
        """Return the agent_id for *namespace*, creating one if needed."""
        if namespace in self._agents:
            return self._agents[namespace]

        agent_name = f"mem-bench-{namespace}"

        # Check if agent already exists by listing agents
        try:
            resp = requests.get(
                f"{self._base_url}/v1/agents",
                headers=self._headers(),
                params={"name": agent_name},
                timeout=30,
            )
            resp.raise_for_status()
            agents = resp.json()

            # Search for existing agent with matching name
            if isinstance(agents, list):
                for agent in agents:
                    if agent.get("name") == agent_name:
                        agent_id = agent["id"]
                        self._agents[namespace] = agent_id
                        return agent_id
        except Exception:
            logger.debug("Could not list agents, will attempt to create", exc_info=True)

        # Create a new agent
        resp = requests.post(
            f"{self._base_url}/v1/agents/",
            headers=self._headers(),
            json={
                "name": agent_name,
                "model": "letta/letta-free",
                "embedding_model": "letta/letta-free",
                "description": f"mem-bench evaluation agent for namespace '{namespace}'",
            },
            timeout=60,
        )
        resp.raise_for_status()
        agent_id = resp.json()["id"]
        self._agents[namespace] = agent_id
        return agent_id

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        agent_id = self._ensure_agent(namespace)

        for item in items:
            # Embed document_id in the text so we can recover it during recall
            text = f"[document_id:{item.document_id}]\n{item.content}"
            if item.timestamp:
                text = f"[{item.timestamp}] {text}"

            try:
                resp = requests.post(
                    f"{self._base_url}/v1/agents/{agent_id}/archival-memory",
                    headers=self._headers(),
                    json={"text": text},
                    timeout=60,
                )
                resp.raise_for_status()
            except Exception:
                logger.exception(
                    "Letta ingest failed for document_id=%s", item.document_id
                )
                raise

    def recall(
        self, query: RecallQuery, *, namespace: str = "default"
    ) -> list[RecallResult]:
        if namespace not in self._agents:
            # No agent has been created yet for this namespace
            return []

        agent_id = self._agents[namespace]

        try:
            resp = requests.get(
                f"{self._base_url}/v1/agents/{agent_id}/archival-memory",
                headers=self._headers(),
                params={"query": query.query, "limit": query.top_k},
                timeout=60,
            )
            resp.raise_for_status()
            passages = resp.json()
        except Exception:
            logger.exception("Letta recall failed")
            raise

        if isinstance(passages, dict):
            passages = passages.get("passages", [])

        results: list[RecallResult] = []
        for i, passage in enumerate(passages):
            content = passage.get("text", "") or ""
            metadata = passage.get("metadata", {}) or {}

            # Extract document_id from embedded tag
            doc_id_match = re.search(r"\[document_id:([^\]]+)\]", content)
            if doc_id_match:
                doc_id = doc_id_match.group(1)
                # Strip the tag from content
                content = re.sub(r"\[document_id:[^\]]+\]\n?", "", content)
            else:
                doc_id = passage.get("id", f"passage-{i}")

            # Use inverse rank as score proxy
            score = 1.0 / (i + 1)

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
        if namespace not in self._agents:
            return

        agent_id = self._agents.pop(namespace)

        try:
            resp = requests.delete(
                f"{self._base_url}/v1/agents/{agent_id}",
                headers=self._headers(),
                timeout=30,
            )
            if resp.status_code not in (200, 204, 404):
                logger.warning(
                    "Letta agent deletion returned %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:
            logger.warning(
                "Letta cleanup failed for namespace=%s", namespace, exc_info=True
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Letta"

    @property
    def capabilities(self) -> set[str]:
        return {"tiered"}
