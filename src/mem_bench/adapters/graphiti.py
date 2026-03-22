"""Graphiti (by Zep) knowledge-graph memory adapter.

Graphiti is an async-native library built on Neo4j.  This adapter wraps its
async API using ``asyncio.run()`` so it can satisfy the synchronous
``MemoryAdapter`` protocol.

Install with: pip install mem-bench[graphiti]
Requires a running Neo4j instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an existing event loop (e.g. Jupyter).  Create a new
        # loop in a thread to avoid ``RuntimeError: This event loop is already
        # running``.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.new_event_loop).result()

    return asyncio.new_event_loop()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class GraphitiAdapter(BaseAdapter):
    """Adapter for the Graphiti knowledge-graph memory system.

    **Requires OpenAI API access.** Graphiti uses OpenAI models internally for
    entity extraction and graph construction. Ensure ``OPENAI_API_KEY`` is set
    in your environment or pass it via the ``openai_api_key`` parameter.

    Args:
        uri: Neo4j connection URI.  Defaults to ``NEO4J_URI`` env var or
             ``bolt://localhost:7687``.
        user: Neo4j username.  Defaults to ``NEO4J_USER`` or ``neo4j``.
        password: Neo4j password.  Defaults to ``NEO4J_PASSWORD`` or ``password``.
        openai_api_key: OpenAI API key.  If provided, it will be set as the
            ``OPENAI_API_KEY`` environment variable for this process.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        openai_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set OpenAI API key in env if provided explicitly.
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self._graphiti: Any = None  # lazy

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_graphiti(self) -> Any:
        if self._graphiti is not None:
            return self._graphiti

        try:
            from graphiti_core import Graphiti  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "graphiti-core is not installed.  Install it with:\n"
                "  pip install mem-bench[graphiti]\n"
                "or:\n"
                "  pip install graphiti-core\n\n"
                "You also need a running Neo4j instance."
            ) from exc

        self._graphiti = Graphiti(self._uri, self._user, self._password)
        return self._graphiti

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Graphiti requires the OpenAI API. Set the OPENAI_API_KEY "
                "environment variable or pass openai_api_key to the adapter."
            )

        graphiti = self._get_graphiti()

        async def _ingest() -> None:
            for item in items:
                ref_time = datetime.now(timezone.utc)
                if item.timestamp:
                    try:
                        ref_time = datetime.fromisoformat(item.timestamp)
                        if ref_time.tzinfo is None:
                            ref_time = ref_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass

                await graphiti.add_episode(
                    name=item.document_id,
                    episode_body=item.content,
                    source_description=f"mem-bench ingest (namespace={namespace})",
                    reference_time=ref_time,
                    group_id=namespace,
                )

        _run_async(_ingest())

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        graphiti = self._get_graphiti()

        async def _recall() -> list[RecallResult]:
            search_results = await graphiti.search(
                query=query.query,
                group_ids=[namespace],
                num_results=query.top_k,
            )

            results: list[RecallResult] = []
            for i, edge in enumerate(search_results):
                # Graphiti search returns edge objects with fact, source_node, etc.
                content = getattr(edge, "fact", "") or str(edge)
                doc_id = getattr(edge, "uuid", "") or f"edge-{i}"
                metadata: dict[str, Any] = {}

                # Extract available metadata from the edge
                for attr in ("created_at", "source_node", "target_node", "episodes"):
                    val = getattr(edge, attr, None)
                    if val is not None:
                        metadata[attr] = str(val)

                # Graphiti doesn't return numeric scores for all search types;
                # use inverse rank as a proxy.
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

        return _run_async(_recall())

    def cleanup(self, *, namespace: str = "default") -> None:
        graphiti = self._get_graphiti()

        async def _cleanup() -> None:
            try:
                # Graphiti provides a build_indices_and_constraints helper;
                # for cleanup, use the underlying Neo4j driver to delete
                # nodes/edges scoped to the group_id (namespace).
                driver = graphiti.driver
                async with driver.session() as session:
                    # Delete all relationships and nodes with matching group_id
                    await session.run(
                        "MATCH (n {group_id: $gid}) DETACH DELETE n",
                        {"gid": namespace},
                    )
            except Exception:
                logger.warning("Graphiti cleanup failed for namespace=%s", namespace, exc_info=True)

        _run_async(_cleanup())

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Graphiti"

    @property
    def capabilities(self) -> set[str]:
        return {"graph", "temporal"}
