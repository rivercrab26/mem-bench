"""Benchmark registry used by CLI commands."""

from __future__ import annotations

from mem_bench.benchmarks.halumem import HaluMemBenchmark
from mem_bench.benchmarks.locomo import LoCoMoBenchmark
from mem_bench.benchmarks.longmemeval import LongMemEvalBenchmark
from mem_bench.core.benchmark import Benchmark

_BENCHMARKS: dict[str, type] = {
    "longmemeval": LongMemEvalBenchmark,
    "locomo": LoCoMoBenchmark,
    "halumem": HaluMemBenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    """Instantiate a benchmark by name.

    Raises:
        ValueError: If the benchmark name is not registered.
    """
    if name not in _BENCHMARKS:
        available = list(_BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return _BENCHMARKS[name]()


def list_benchmarks() -> list[str]:
    """Return sorted list of registered benchmark names."""
    return sorted(_BENCHMARKS.keys())
