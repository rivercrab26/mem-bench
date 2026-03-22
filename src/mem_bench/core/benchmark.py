"""Benchmark interface and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

from mem_bench.core.types import IngestItem


@dataclass
class BenchmarkSample:
    """A single evaluation instance from a benchmark dataset."""

    sample_id: str
    question: str
    reference_answer: str
    question_type: str
    ingest_items: list[IngestItem]
    ground_truth_doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Benchmark(Protocol):
    """Interface for loading and iterating over a benchmark dataset."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        """Download (if needed) and load the dataset."""
        ...

    def __iter__(self) -> Iterator[BenchmarkSample]: ...

    def __len__(self) -> int: ...
