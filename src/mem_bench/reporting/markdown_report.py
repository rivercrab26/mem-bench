"""Markdown report generation."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from mem_bench.core.runner import RunResult
from mem_bench.core.types import SampleResult

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_by_question_type(
    samples: list[SampleResult],
) -> dict[str, list[SampleResult]]:
    groups: dict[str, list[SampleResult]] = defaultdict(list)
    for s in samples:
        groups[s.question_type].append(s)
    return dict(groups)


def _metric_keys(samples: list[SampleResult]) -> list[str]:
    keys: set[str] = set()
    for s in samples:
        keys.update(s.retrieval_metrics.keys())
    return sorted(keys)


def save_markdown_report(run_result: RunResult, output_dir: str | Path) -> Path:
    """Generate a Markdown report and save it to *output_dir/report.md*.

    Args:
        run_result: The completed benchmark run result.
        output_dir: Directory to write the report into (created if needed).

    Returns:
        Path to the generated ``report.md`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # -- Title ----------------------------------------------------------------
    lines.append(f"# Benchmark Report: {run_result.adapter_name}")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Benchmark | {run_result.benchmark_name} |")
    lines.append(f"| Split | {run_result.split} |")
    lines.append(f"| Samples | {run_result.num_samples} |")
    lines.append(f"| Failed | {run_result.num_failed} |")
    lines.append(f"| Total Time | {run_result.total_seconds:.1f}s |")
    lines.append("")

    # -- Metrics by question type ---------------------------------------------
    groups = _group_by_question_type(run_result.sample_results)
    all_keys = _metric_keys(run_result.sample_results)

    if all_keys:
        lines.append("## Metrics by Question Type")
        lines.append("")

        header = "| Question Type | Count | " + " | ".join(all_keys) + " |"
        sep = "|---|---:|" + "|".join(["---:" for _ in all_keys]) + "|"
        lines.append(header)
        lines.append(sep)

        for qtype in sorted(groups.keys()):
            samples = groups[qtype]
            cells = [qtype, str(len(samples))]
            for key in all_keys:
                vals = [s.retrieval_metrics.get(key, 0.0) for s in samples]
                cells.append(f"{_mean(vals):.4f}")
            lines.append("| " + " | ".join(cells) + " |")

        # Overall
        all_samples = run_result.sample_results
        cells = ["**Overall**", str(len(all_samples))]
        for key in all_keys:
            vals = [s.retrieval_metrics.get(key, 0.0) for s in all_samples]
            cells.append(f"**{_mean(vals):.4f}**")
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # -- Aggregate metrics (flat) ---------------------------------------------
    agg = run_result.aggregate_metrics
    if agg:
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for key in sorted(agg.keys()):
            lines.append(f"| {key} | {agg[key]:.4f} |")
        lines.append("")

    # -- Config ---------------------------------------------------------------
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")

    import json

    lines.append(json.dumps(run_result.config, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # -- Write ----------------------------------------------------------------
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown report to %s", report_path)

    return report_path
