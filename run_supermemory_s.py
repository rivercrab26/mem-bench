#!/usr/bin/env python3
"""Run Supermemory on LongMemEval S split with stratified sampling.

Takes 8 questions per type (48 total) to cover all 6 question types
while keeping total runtime manageable (~4 hours).
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mem_bench.adapters.supermemory import SupermemoryAdapter
from mem_bench.benchmarks.longmemeval import LongMemEvalBenchmark
from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.config import (
    AdapterConfig,
    JudgeConfig,
    MetricsConfig,
    ReportingConfig,
    RunConfig,
)
from mem_bench.core.runner import BenchmarkRunner


class StratifiedBenchmark:
    """Wrapper that stratified-samples from a benchmark."""

    def __init__(self, inner: LongMemEvalBenchmark, per_type: int = 8):
        self._inner = inner
        self._per_type = per_type
        self._samples: list[BenchmarkSample] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    def load(self, *, split: str = "s", limit: int | None = None) -> None:
        self._inner.load(split=split, limit=None)  # Load all
        # Group by type
        by_type: dict[str, list[BenchmarkSample]] = defaultdict(list)
        for s in self._inner:
            by_type[s.question_type].append(s)
        # Sample per_type from each
        self._samples = []
        for qtype in sorted(by_type.keys()):
            items = by_type[qtype]
            self._samples.extend(items[: self._per_type])
        print(f"Stratified sample: {len(self._samples)} questions "
              f"({self._per_type} per type, {len(by_type)} types)")

    def __iter__(self):
        return iter(self._samples)

    def __len__(self):
        return len(self._samples)


def main():
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        print("Set SUPERMEMORY_API_KEY env var")
        sys.exit(1)

    config = RunConfig(
        benchmark="longmemeval",
        split="s",
        limit=0,
        output_dir="./full_test_results/supermemory_s_stratified",
        adapter=AdapterConfig(name="supermemory"),
        judge=JudgeConfig(
            enabled=bool(openai_key),
            model="openai/gpt-4o-mini",
            provider="openai",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        metrics=MetricsConfig(retrieval_k=[1, 3, 5, 10]),
        reporting=ReportingConfig(formats=["console", "json", "markdown", "html"]),
    )

    adapter = SupermemoryAdapter(api_key=api_key)
    benchmark = StratifiedBenchmark(LongMemEvalBenchmark(), per_type=8)
    benchmark.load(split="s")

    runner = BenchmarkRunner(adapter=adapter, benchmark=benchmark, config=config)
    result = runner.run()

    # Print summary
    m = result.aggregate_metrics
    print(f"\nSupermemory S-split stratified results:")
    print(f"  Samples: {result.num_samples}, Failed: {result.num_failed}")
    print(f"  QA accuracy: {m.get('qa_accuracy', 'N/A')}")
    print(f"  recall_any@1: {m.get('recall_any@1', 'N/A')}")


if __name__ == "__main__":
    main()
