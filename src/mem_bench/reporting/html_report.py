"""Self-contained HTML report generation with inline CSS and SVG charts."""

from __future__ import annotations

import json
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


def _qa_accuracy_for(samples: list[SampleResult]) -> float | None:
    scores = [s.qa_score for s in samples if s.qa_score is not None]
    if not scores:
        return None
    return sum(1 for s in scores if s > 0.5) / len(scores)


def _color_for_value(value: float) -> str:
    """Return a background color based on metric value (0-1 scale)."""
    if value >= 0.8:
        return "#d4edda"  # green
    elif value >= 0.5:
        return "#fff3cd"  # yellow
    else:
        return "#f8d7da"  # red


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _build_svg_bar_chart(
    groups: dict[str, list[SampleResult]],
    metric_key: str,
    width: int = 600,
    bar_height: int = 28,
    margin_left: int = 160,
) -> str:
    """Build a simple horizontal SVG bar chart for a given metric across question types."""
    sorted_types = sorted(groups.keys())
    n = len(sorted_types)
    if n == 0:
        return ""

    gap = 6
    chart_height = n * (bar_height + gap) + 20
    max_bar_width = width - margin_left - 60

    values: list[float] = []
    for qtype in sorted_types:
        samples = groups[qtype]
        if metric_key == "qa_accuracy":
            v = _qa_accuracy_for(samples)
            values.append(v if v is not None else 0.0)
        else:
            vals = [s.retrieval_metrics.get(metric_key, 0.0) for s in samples]
            values.append(_mean(vals))

    max_val = max(values) if values else 1.0
    if max_val == 0:
        max_val = 1.0
    # Clamp scale to at least 1.0 for ratio metrics
    scale = min(max_val, 1.0) if max_val <= 1.0 else max_val

    lines: list[str] = []
    lines.append(f'<svg width="{width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">')
    lines.append('<style>text { font-family: system-ui, sans-serif; font-size: 13px; }</style>')

    for i, (qtype, val) in enumerate(zip(sorted_types, values)):
        y = i * (bar_height + gap) + 10
        bar_w = int((val / scale) * max_bar_width) if scale > 0 else 0
        bar_w = max(bar_w, 2)  # minimum visible width
        color = _color_for_value(val) if scale <= 1.0 else "#6fa8dc"

        # Label
        lines.append(f'<text x="{margin_left - 8}" y="{y + bar_height // 2 + 5}" text-anchor="end" fill="#333">{_escape_html(qtype)}</text>')
        # Bar
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_height}" fill="{color}" rx="3" />')
        # Value text
        lines.append(f'<text x="{margin_left + bar_w + 6}" y="{y + bar_height // 2 + 5}" fill="#333">{val:.4f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def save_html_report(run_result: RunResult, output_dir: str | Path) -> Path:
    """Generate a self-contained HTML report and save it to output_dir/report.html.

    Args:
        run_result: The completed benchmark run result.
        output_dir: Directory to write the report into (created if needed).

    Returns:
        Path to the generated ``report.html`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_by_question_type(run_result.sample_results)
    all_keys = _metric_keys(run_result.sample_results)
    has_qa = any(s.qa_score is not None for s in run_result.sample_results)
    agg = run_result.aggregate_metrics

    column_keys = list(all_keys)
    if has_qa:
        column_keys.append("qa_accuracy")

    # Pick a representative recall metric for the bar chart
    recall_chart_keys = [k for k in column_keys if k.startswith("recall_") or k == "qa_accuracy"]
    if not recall_chart_keys and column_keys:
        recall_chart_keys = column_keys[:3]

    # Build metadata dict (may or may not exist on RunResult)
    metadata = getattr(run_result, "metadata", {}) or {}

    html_parts: list[str] = []

    # --- Header ---
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark Report: {_escape_html(run_result.adapter_name)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; color: #333; padding: 24px; line-height: 1.5; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 8px; color: #1a1a2e; }}
  h2 {{ font-size: 1.2rem; margin: 24px 0 12px 0; color: #16213e; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
  .summary-card {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .summary-card .label {{ font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .summary-card .value {{ font-size: 1.3rem; font-weight: 600; color: #1a1a2e; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 12px 0; }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left; font-size: 0.85rem; font-weight: 600; }}
  th.num {{ text-align: right; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  td.num {{ text-align: right; font-family: 'SF Mono', 'Consolas', monospace; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.overall td {{ font-weight: 700; background: #f0f4ff; }}
  .chart-container {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 12px 0; overflow-x: auto; }}
  pre {{ background: #1a1a2e; color: #e0e0e0; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; line-height: 1.6; }}
  .timing-table td, .timing-table th {{ font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">
<h1>Benchmark Report: {_escape_html(run_result.adapter_name)}</h1>
""")

    # --- Summary cards ---
    html_parts.append('<div class="summary-grid">')
    cards = [
        ("Benchmark", run_result.benchmark_name),
        ("Split", run_result.split),
        ("Samples", str(run_result.num_samples)),
        ("Failed", str(run_result.num_failed)),
        ("Total Time", f"{run_result.total_seconds:.1f}s"),
    ]
    if metadata.get("mem_bench_version"):
        cards.append(("Version", metadata["mem_bench_version"]))
    if metadata.get("timestamp"):
        cards.append(("Timestamp", metadata["timestamp"][:19]))

    for label, value in cards:
        html_parts.append(f'<div class="summary-card"><div class="label">{_escape_html(label)}</div><div class="value">{_escape_html(value)}</div></div>')
    html_parts.append('</div>')

    # --- Metrics table ---
    if column_keys:
        html_parts.append('<h2>Metrics by Question Type</h2>')
        html_parts.append('<table><thead><tr><th>Question Type</th><th class="num">Count</th>')
        for k in column_keys:
            html_parts.append(f'<th class="num">{_escape_html(k)}</th>')
        html_parts.append('</tr></thead><tbody>')

        for qtype in sorted(groups.keys()):
            samples = groups[qtype]
            html_parts.append(f'<tr><td>{_escape_html(qtype)}</td><td class="num">{len(samples)}</td>')
            for key in column_keys:
                if key == "qa_accuracy":
                    v = _qa_accuracy_for(samples)
                    if v is not None:
                        bg = _color_for_value(v)
                        html_parts.append(f'<td class="num" style="background:{bg}">{v:.4f}</td>')
                    else:
                        html_parts.append('<td class="num">-</td>')
                else:
                    vals = [s.retrieval_metrics.get(key, 0.0) for s in samples]
                    v = _mean(vals)
                    bg = _color_for_value(v)
                    html_parts.append(f'<td class="num" style="background:{bg}">{v:.4f}</td>')
            html_parts.append('</tr>')

        # Overall row
        all_samples = run_result.sample_results
        html_parts.append(f'<tr class="overall"><td>Overall</td><td class="num">{len(all_samples)}</td>')
        for key in column_keys:
            if key == "qa_accuracy":
                v = _qa_accuracy_for(all_samples)
                if v is not None:
                    bg = _color_for_value(v)
                    html_parts.append(f'<td class="num" style="background:{bg}">{v:.4f}</td>')
                else:
                    html_parts.append('<td class="num">-</td>')
            else:
                vals = [s.retrieval_metrics.get(key, 0.0) for s in all_samples]
                v = _mean(vals)
                bg = _color_for_value(v)
                html_parts.append(f'<td class="num" style="background:{bg}">{v:.4f}</td>')
        html_parts.append('</tr>')

        html_parts.append('</tbody></table>')

    # --- SVG bar charts ---
    if recall_chart_keys and groups:
        html_parts.append('<h2>Recall by Question Type</h2>')
        for chart_key in recall_chart_keys[:4]:
            html_parts.append(f'<div class="chart-container"><strong>{_escape_html(chart_key)}</strong><br>')
            html_parts.append(_build_svg_bar_chart(groups, chart_key))
            html_parts.append('</div>')

    # --- Timing summary ---
    timing_keys = [k for k in sorted(agg.keys()) if k.endswith("_seconds")]
    if timing_keys:
        html_parts.append('<h2>Timing Summary</h2>')
        html_parts.append('<table class="timing-table"><thead><tr><th>Metric</th><th class="num">Value (s)</th></tr></thead><tbody>')
        for key in timing_keys:
            html_parts.append(f'<tr><td>{_escape_html(key)}</td><td class="num">{agg[key]:.4f}</td></tr>')
        html_parts.append('</tbody></table>')

    # --- Config dump ---
    html_parts.append('<h2>Configuration</h2>')
    config_json = json.dumps(run_result.config, indent=2, default=str)
    html_parts.append(f'<pre>{_escape_html(config_json)}</pre>')

    # --- Metadata ---
    if metadata:
        html_parts.append('<h2>Reproducibility Metadata</h2>')
        meta_json = json.dumps(metadata, indent=2, default=str)
        html_parts.append(f'<pre>{_escape_html(meta_json)}</pre>')

    # --- Footer ---
    html_parts.append('</div></body></html>')

    report_path = output_dir / "report.html"
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Wrote HTML report to %s", report_path)

    return report_path
