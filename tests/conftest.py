"""Shared fixtures for mem-bench tests."""

from __future__ import annotations

import pytest

from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem, RecallQuery


@pytest.fixture()
def sample_items() -> list[IngestItem]:
    """Five simple IngestItems for testing."""
    return [
        IngestItem(content="The capital of France is Paris.", document_id="doc_1"),
        IngestItem(content="Python is a popular programming language.", document_id="doc_2"),
        IngestItem(content="The Great Wall of China is visible from space.", document_id="doc_3"),
        IngestItem(content="Water boils at 100 degrees Celsius.", document_id="doc_4"),
        IngestItem(
            content="Albert Einstein developed the theory of relativity.", document_id="doc_5"
        ),
    ]


@pytest.fixture()
def sample_query() -> RecallQuery:
    """A recall query about France."""
    return RecallQuery(query="What is the capital of France?", top_k=3)


@pytest.fixture()
def sample_benchmark_sample(sample_items: list[IngestItem]) -> BenchmarkSample:
    """A benchmark sample using the sample items."""
    return BenchmarkSample(
        sample_id="test_001",
        question="What is the capital of France?",
        reference_answer="Paris",
        question_type="single-session-user",
        ingest_items=sample_items,
        ground_truth_doc_ids=["doc_1"],
    )
