"""Memory adapter interface.

Third-party memory systems implement the MemoryAdapter protocol.
Uses Protocol (PEP 544) so adapters never need to import mem-bench.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from mem_bench.core.types import IngestItem, RecallQuery, RecallResult


@runtime_checkable
class MemoryAdapter(Protocol):
    """Minimal interface every memory system must implement.

    Three methods. That's it.
    """

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        """Store items into the memory system.

        Args:
            items: Batch of content items to ingest.
            namespace: Isolation scope (bank/collection/user_id).
        """
        ...

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        """Retrieve relevant memories, ordered by relevance (best first).

        Args:
            query: The recall query with text and parameters.
            namespace: Same isolation scope used during ingest.
        """
        ...

    def cleanup(self, *, namespace: str = "default") -> None:
        """Delete all data in the given namespace."""
        ...


class BaseAdapter:
    """Optional convenience base class. Protocol is sufficient; this is not required."""

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        raise NotImplementedError

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        raise NotImplementedError

    def cleanup(self, *, namespace: str = "default") -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def capabilities(self) -> set[str]:
        """Declare adapter capabilities: temporal, graph, tiered, user_facts, async."""
        return set()
