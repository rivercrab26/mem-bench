"""Tests for the BenchmarkRunner."""

from __future__ import annotations

from typing import Iterator

from mem_bench.adapters.bm25 import BM25Adapter
from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.config import RunConfig
from mem_bench.core.runner import BenchmarkRunner, RunResult
from mem_bench.core.types import IngestItem


class _MockBenchmark:
    """A tiny in-memory benchmark with 3 samples for testing."""

    def __init__(self, samples: list[BenchmarkSample] | None = None):
        self._samples = samples or self._default_samples()

    @property
    def name(self) -> str:
        return "mock_bench"

    @property
    def version(self) -> str:
        return "0.1"

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        pass

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _default_samples() -> list[BenchmarkSample]:
        items_1 = [
            IngestItem(content="Paris is the capital of France.", document_id="d1"),
            IngestItem(content="Berlin is the capital of Germany.", document_id="d2"),
        ]
        items_2 = [
            IngestItem(content="Python was created by Guido van Rossum.", document_id="d3"),
            IngestItem(content="Java was created by James Gosling.", document_id="d4"),
        ]
        items_3 = [
            IngestItem(content="The sun rises in the east.", document_id="d5"),
            IngestItem(content="Water freezes at 0 degrees.", document_id="d6"),
        ]
        return [
            BenchmarkSample(
                sample_id="s1",
                question="What is the capital of France?",
                reference_answer="Paris",
                question_type="single-session-user",
                ingest_items=items_1,
                ground_truth_doc_ids=["d1"],
            ),
            BenchmarkSample(
                sample_id="s2",
                question="Who created Python?",
                reference_answer="Guido van Rossum",
                question_type="single-session-user",
                ingest_items=items_2,
                ground_truth_doc_ids=["d3"],
            ),
            BenchmarkSample(
                sample_id="s3",
                question="What temperature does water freeze at?",
                reference_answer="0 degrees Celsius",
                question_type="temporal-reasoning",
                ingest_items=items_3,
                ground_truth_doc_ids=["d6"],
            ),
        ]


class TestBenchmarkRunner:
    """Test BenchmarkRunner with BM25 and a mock benchmark."""

    def _make_runner(self, samples: list[BenchmarkSample] | None = None) -> BenchmarkRunner:
        adapter = BM25Adapter()
        benchmark = _MockBenchmark(samples)
        config = RunConfig(
            benchmark="mock_bench",
            split="test",
            metrics={"retrieval_k": [1, 3, 5]},
        )
        return BenchmarkRunner(adapter, benchmark, config)

    def test_run_returns_run_result(self):
        runner = self._make_runner()
        result = runner.run()
        assert isinstance(result, RunResult)

    def test_correct_sample_count(self):
        runner = self._make_runner()
        result = runner.run()
        assert result.num_samples == 3
        assert len(result.sample_results) == 3

    def test_aggregate_metrics_present(self):
        runner = self._make_runner()
        result = runner.run()
        agg = result.aggregate_metrics

        assert "recall_any@1" in agg
        assert "mrr" in agg
        assert "mean_ingest_seconds" in agg
        assert "mean_recall_seconds" in agg

    def test_all_samples_have_metrics(self):
        runner = self._make_runner()
        result = runner.run()

        for sr in result.sample_results:
            assert "recall_any@1" in sr.retrieval_metrics
            assert "mrr" in sr.retrieval_metrics

    def test_no_failures_on_valid_data(self):
        runner = self._make_runner()
        result = runner.run()
        assert result.num_failed == 0

    def test_per_sample_namespace_isolation(self):
        """Each sample uses its own namespace so data does not leak."""
        items_shared = [
            IngestItem(content="Overlap content about France and Paris.", document_id="shared_1"),
        ]
        samples = [
            BenchmarkSample(
                sample_id="iso_1",
                question="France?",
                reference_answer="Paris",
                question_type="single-session-user",
                ingest_items=items_shared,
                ground_truth_doc_ids=["shared_1"],
            ),
            BenchmarkSample(
                sample_id="iso_2",
                question="Germany?",
                reference_answer="Berlin",
                question_type="single-session-user",
                ingest_items=[
                    IngestItem(content="Berlin is the capital of Germany.", document_id="de_1"),
                ],
                ground_truth_doc_ids=["de_1"],
            ),
        ]
        runner = self._make_runner(samples)
        result = runner.run()

        # Both should succeed with their own data; no cross-contamination.
        assert result.num_failed == 0
        assert result.num_samples == 2


class TestFailedSamples:
    """Test that failed samples are counted."""

    def test_failed_sample_counted(self):
        """A sample that causes an error should be counted as failed."""

        class _FailingAdapter:
            name = "failing"
            _call_count = 0

            def ingest(self, items, *, namespace="default"):
                self._call_count += 1
                if self._call_count <= 3:
                    # Fail for first sample (warmup cleanup + pre-cleanup + ingest = first 3 calls concern sample 1 ingest)
                    raise RuntimeError("Simulated failure")

            def recall(self, query, *, namespace="default"):
                return []

            def cleanup(self, *, namespace="default"):
                pass

        adapter = _FailingAdapter()
        benchmark = _MockBenchmark()
        config = RunConfig(benchmark="mock_bench", split="test")
        runner = BenchmarkRunner(adapter, benchmark, config)
        result = runner.run()

        # At least the first sample should fail due to ingest errors.
        assert result.num_failed >= 1
