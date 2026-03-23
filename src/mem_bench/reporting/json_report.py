"""JSON report generation: results.jsonl + summary.json."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mem_bench.core.runner import RunResult
from mem_bench.reporting._utils import _FACT_EXTRACTION_NOTE, detect_fact_extraction_mode

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Any:
    """JSON-safe serialisation for dataclass fields."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json_report(run_result: RunResult, output_dir: str | Path) -> Path:
    """Save benchmark results as JSON files.

    Creates two files inside *output_dir*:

    * ``results.jsonl`` -- one JSON object per sample (streaming-friendly).
    * ``summary.json``  -- aggregate metrics and run metadata.

    Args:
        run_result: The completed benchmark run result.
        output_dir: Directory to write files into (created if needed).

    Returns:
        Path to *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- results.jsonl --------------------------------------------------------
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for sample in run_result.sample_results:
            record = asdict(sample)
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    logger.info("Wrote %d sample results to %s", len(run_result.sample_results), results_path)

    # -- summary.json ---------------------------------------------------------
    summary: dict[str, Any] = {
        "benchmark_name": run_result.benchmark_name,
        "split": run_result.split,
        "adapter_name": run_result.adapter_name,
        "num_samples": run_result.num_samples,
        "num_failed": run_result.num_failed,
        "total_seconds": run_result.total_seconds,
        "aggregate_metrics": run_result.aggregate_metrics,
        "config": run_result.config,
        "metadata": run_result.metadata,
    }
    if detect_fact_extraction_mode(run_result.sample_results):
        summary["fact_extraction_mode"] = True
        summary["fact_extraction_note"] = _FACT_EXTRACTION_NOTE
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote summary to %s", summary_path)

    return output_dir
