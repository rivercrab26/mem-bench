"""Example: Writing a custom benchmark for mem-bench."""

from __future__ import annotations

from typing import Iterator

from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem


class MyBenchmark:
    """A minimal benchmark with 3 hardcoded questions."""

    @property
    def name(self) -> str:
        return "my-benchmark"

    @property
    def version(self) -> str:
        return "1.0"

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        self._samples = [
            BenchmarkSample(
                sample_id="q1",
                question="What is Alice's favorite color?",
                reference_answer="blue",
                question_type="single-hop",
                ingest_items=[
                    IngestItem(content="Alice told me her favorite color is blue.", document_id="s1"),
                    IngestItem(content="Bob likes hiking in the mountains.", document_id="s2"),
                ],
                ground_truth_doc_ids=["s1"],
            ),
            BenchmarkSample(
                sample_id="q2",
                question="Where does Bob like to hike?",
                reference_answer="mountains",
                question_type="single-hop",
                ingest_items=[
                    IngestItem(content="Alice told me her favorite color is blue.", document_id="s1"),
                    IngestItem(content="Bob likes hiking in the mountains.", document_id="s2"),
                ],
                ground_truth_doc_ids=["s2"],
            ),
            BenchmarkSample(
                sample_id="q3_abs",
                question="What is Charlie's phone number?",
                reference_answer="Not mentioned in any conversation.",
                question_type="single-hop",
                ingest_items=[
                    IngestItem(content="Alice and Bob discussed the weather.", document_id="s3"),
                ],
                ground_truth_doc_ids=[],
            ),
        ]
        if limit:
            self._samples = self._samples[:limit]

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


# Usage:
#   mem-bench run --adapter bm25 --benchmark examples.custom_benchmark:MyBenchmark
