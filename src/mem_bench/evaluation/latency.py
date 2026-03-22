"""Latency statistics computation."""

from __future__ import annotations

import statistics
from mem_bench.core.types import SampleResult


def compute_latency_stats(results: list[SampleResult]) -> dict[str, float]:
    """Compute latency statistics across all samples."""
    if not results:
        return {}

    ingest_times = [r.timing.ingest_seconds for r in results]
    recall_times = [r.timing.recall_seconds for r in results]

    stats = {}

    for name, values in [("ingest", ingest_times), ("recall", recall_times)]:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        stats[f"{name}_mean"] = statistics.mean(values)
        stats[f"{name}_median"] = statistics.median(values)
        stats[f"{name}_p95"] = sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[0]
        stats[f"{name}_p99"] = sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[0]
        stats[f"{name}_min"] = min(values)
        stats[f"{name}_max"] = max(values)

    return stats
