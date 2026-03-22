"""Shared data types for mem-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestItem:
    """A single unit of content to store in the memory system."""

    content: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None  # ISO 8601


@dataclass(frozen=True)
class RecallQuery:
    """A query to retrieve relevant memories."""

    query: str
    top_k: int = 10
    metadata_filter: dict[str, Any] | None = None


@dataclass
class RecallResult:
    """A single retrieved memory."""

    document_id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingInfo:
    """Timing measurements for a single benchmark sample."""

    ingest_seconds: float = 0.0
    recall_seconds: float = 0.0
    cleanup_seconds: float = 0.0


@dataclass
class SampleResult:
    """Full result for a single benchmark sample."""

    sample_id: str
    question_type: str
    recall_results: list[RecallResult] = field(default_factory=list)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    hypothesis: str = ""
    qa_score: float | None = None
    timing: TimingInfo = field(default_factory=TimingInfo)
