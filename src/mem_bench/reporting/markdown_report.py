"""Markdown report generation."""

from __future__ import annotations

import logging
from pathlib import Path

from mem_bench.core.runner import RunResult
from mem_bench.reporting._utils import (
    _FACT_EXTRACTION_NOTE,
    _group_by_question_type,
    _mean,
    _metric_keys,
    _qa_accuracy_for,
    detect_fact_extraction_mode,
)

logger = logging.getLogger(__name__)


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
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Benchmark | {run_result.benchmark_name} |")
    lines.append(f"| Split | {run_result.split} |")
    lines.append(f"| Samples | {run_result.num_samples} |")
    lines.append(f"| Failed | {run_result.num_failed} |")
    lines.append(f"| Total Time | {run_result.total_seconds:.1f}s |")
    lines.append("")

    # -- Metrics by question type ---------------------------------------------
    groups = _group_by_question_type(run_result.sample_results)
    all_keys = _metric_keys(run_result.sample_results)
    has_qa = any(s.qa_score is not None for s in run_result.sample_results)

    is_fact_extraction = detect_fact_extraction_mode(run_result.sample_results)
    if is_fact_extraction:
        lines.append(f"> **NOTE:** {_FACT_EXTRACTION_NOTE}")
        lines.append("")

    if all_keys or has_qa:
        lines.append("## Metrics by Question Type")
        lines.append("")

        column_keys = list(all_keys)
        if has_qa:
            column_keys.append("qa_accuracy")

        header = "| Question Type | Count | " + " | ".join(column_keys) + " |"
        sep = "|---|---:|" + "|".join(["---:" for _ in column_keys]) + "|"
        lines.append(header)
        lines.append(sep)

        for qtype in sorted(groups.keys()):
            samples = groups[qtype]
            cells = [qtype, str(len(samples))]
            for key in column_keys:
                if key == "qa_accuracy":
                    qa_acc = _qa_accuracy_for(samples)
                    cells.append(f"{qa_acc:.4f}" if qa_acc is not None else "-")
                else:
                    vals = [s.retrieval_metrics.get(key, 0.0) for s in samples]
                    cells.append(f"{_mean(vals):.4f}")
            lines.append("| " + " | ".join(cells) + " |")

        # Overall
        all_samples = run_result.sample_results
        cells = ["**Overall**", str(len(all_samples))]
        for key in column_keys:
            if key == "qa_accuracy":
                qa_acc = _qa_accuracy_for(all_samples)
                cells.append(f"**{qa_acc:.4f}**" if qa_acc is not None else "-")
            else:
                vals = [s.retrieval_metrics.get(key, 0.0) for s in all_samples]
                cells.append(f"**{_mean(vals):.4f}**")
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # -- Aggregate metrics (flat) ---------------------------------------------
    agg = run_result.aggregate_metrics
    non_timing_keys = [k for k in sorted(agg.keys()) if not k.endswith("_seconds")]
    if non_timing_keys:
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for key in non_timing_keys:
            lines.append(f"| {key} | {agg[key]:.4f} |")
        lines.append("")

    # -- Timing summary -------------------------------------------------------
    timing_keys = [k for k in sorted(agg.keys()) if k.endswith("_seconds")]
    if timing_keys:
        lines.append("## Timing Summary")
        lines.append("")
        lines.append("| Metric | Value (s) |")
        lines.append("|--------|----------:|")
        for key in timing_keys:
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
