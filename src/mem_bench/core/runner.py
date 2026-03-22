"""Benchmark runner.

Orchestrates ingestion, recall, and evaluation for every sample in a
benchmark, collecting per-sample and aggregate results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from mem_bench.core.adapter import MemoryAdapter
from mem_bench.core.benchmark import Benchmark, BenchmarkSample
from mem_bench.core.config import RunConfig
from mem_bench.core.types import (
    RecallQuery,
    RecallResult,
    SampleResult,
    TimingInfo,
)
from mem_bench.evaluation.retrieval import compute_retrieval_metrics

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Aggregate results for a full benchmark run."""

    benchmark_name: str = ""
    split: str = ""
    adapter_name: str = ""
    num_samples: int = 0
    num_failed: int = 0
    sample_results: list[SampleResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Drives a ``MemoryAdapter`` through a ``Benchmark`` and collects metrics.

    Usage::

        runner = BenchmarkRunner(adapter, benchmark, config)
        result = runner.run()
    """

    def __init__(
        self,
        adapter: MemoryAdapter,
        benchmark: Benchmark,
        config: RunConfig,
    ) -> None:
        self.adapter = adapter
        self.benchmark = benchmark
        self.config = config

        # Derive a unique namespace per run to avoid cross-contamination.
        self._namespace = f"mem_bench_{benchmark.name}_{config.split}"

    # -- Public API -----------------------------------------------------------

    def run(self) -> RunResult:
        """Execute the full benchmark and return aggregated results."""
        run_start = time.perf_counter()

        k_values = self.config.metrics.retrieval_k
        top_k = max(k_values) if k_values else 10

        sample_results: list[SampleResult] = []
        num_failed = 0

        adapter_name = getattr(self.adapter, "name", self.adapter.__class__.__name__)

        samples = list(self.benchmark)
        for sample in tqdm(samples, desc=f"Running {adapter_name}", unit="sample"):
            try:
                result = self._run_sample(sample, top_k=top_k, k_values=k_values)
                sample_results.append(result)
            except Exception:
                logger.warning(
                    "Sample %s failed, skipping", sample.sample_id, exc_info=True
                )
                num_failed += 1
                # Record a minimal result so the failure is visible.
                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        question_type=sample.question_type,
                        retrieval_metrics={},
                        timing=TimingInfo(),
                    )
                )

        total_seconds = time.perf_counter() - run_start

        aggregate = self._aggregate_metrics(sample_results)

        return RunResult(
            benchmark_name=self.benchmark.name,
            split=self.config.split,
            adapter_name=adapter_name,
            num_samples=len(sample_results),
            num_failed=num_failed,
            sample_results=sample_results,
            aggregate_metrics=aggregate,
            total_seconds=total_seconds,
            config=self.config.model_dump(),
        )

    # -- Private helpers ------------------------------------------------------

    def _run_sample(
        self,
        sample: BenchmarkSample,
        *,
        top_k: int,
        k_values: list[int],
    ) -> SampleResult:
        """Run a single sample: cleanup -> ingest -> recall -> evaluate -> cleanup."""
        timing = TimingInfo()

        # 1. Pre-cleanup
        t0 = time.perf_counter()
        self.adapter.cleanup(namespace=self._namespace)
        timing.cleanup_seconds = time.perf_counter() - t0

        # 2. Ingest
        t0 = time.perf_counter()
        self.adapter.ingest(sample.ingest_items, namespace=self._namespace)
        timing.ingest_seconds = time.perf_counter() - t0

        # 3. Recall
        query = RecallQuery(
            query=sample.question,
            top_k=top_k,
            metadata_filter=sample.metadata.get("metadata_filter"),
        )
        t0 = time.perf_counter()
        recall_results: list[RecallResult] = self.adapter.recall(
            query, namespace=self._namespace
        )
        timing.recall_seconds = time.perf_counter() - t0

        # 4. Retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            recall_results, sample.ground_truth_doc_ids, k_values=k_values
        )

        # 5. Post-cleanup
        t0 = time.perf_counter()
        self.adapter.cleanup(namespace=self._namespace)
        timing.cleanup_seconds += time.perf_counter() - t0

        return SampleResult(
            sample_id=sample.sample_id,
            question_type=sample.question_type,
            recall_results=recall_results,
            retrieval_metrics=retrieval_metrics,
            timing=timing,
        )

    @staticmethod
    def _aggregate_metrics(results: list[SampleResult]) -> dict[str, float]:
        """Compute mean of each metric across all samples."""
        if not results:
            return {}

        # Collect all metric keys from samples that have metrics.
        metric_accum: dict[str, list[float]] = {}
        timing_accum: dict[str, list[float]] = {
            "ingest_seconds": [],
            "recall_seconds": [],
            "cleanup_seconds": [],
        }

        for r in results:
            for key, val in r.retrieval_metrics.items():
                metric_accum.setdefault(key, []).append(val)
            timing_accum["ingest_seconds"].append(r.timing.ingest_seconds)
            timing_accum["recall_seconds"].append(r.timing.recall_seconds)
            timing_accum["cleanup_seconds"].append(r.timing.cleanup_seconds)

        aggregate: dict[str, float] = {}
        for key, vals in sorted(metric_accum.items()):
            aggregate[key] = sum(vals) / len(vals) if vals else 0.0

        # Timing aggregates
        for key, vals in timing_accum.items():
            if vals:
                aggregate[f"mean_{key}"] = sum(vals) / len(vals)
                aggregate[f"total_{key}"] = sum(vals)

        return aggregate
