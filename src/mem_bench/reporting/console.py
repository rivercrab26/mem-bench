"""Rich console output for benchmark results."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from mem_bench.core.runner import RunResult
from mem_bench.core.types import SampleResult
from mem_bench.reporting._utils import _group_by_question_type, _mean, _qa_accuracy_for

# Key metrics to display (avoids truncation from too many columns).
_DISPLAY_METRICS = [
    "recall_any@1",
    "recall_all@5",
    "ndcg@5",
    "mrr",
    "qa_accuracy",
]


def _pick_display_keys(samples: list[SampleResult]) -> list[str]:
    """Return the subset of display metrics that actually exist in the data."""
    available: set[str] = set()
    for s in samples:
        available.update(s.retrieval_metrics.keys())
    # Also check for qa_accuracy presence via qa_score
    has_qa = any(s.qa_score is not None for s in samples)
    if has_qa:
        available.add("qa_accuracy")
    return [k for k in _DISPLAY_METRICS if k in available]


def print_results(run_result: RunResult) -> None:
    """Print a Rich table summarising a benchmark run.

    Shows per-question-type breakdown and overall aggregate metrics.
    """
    console = Console()
    console.print()

    # -- Header info ----------------------------------------------------------
    console.rule(f"[bold]Results: {run_result.adapter_name} on {run_result.benchmark_name}")
    console.print(
        f"  Split: {run_result.split}  |  "
        f"Samples: {run_result.num_samples}  |  "
        f"Failed: {run_result.num_failed}  |  "
        f"Time: {run_result.total_seconds:.1f}s"
    )
    console.print()

    # -- Per-question-type table ----------------------------------------------
    groups = _group_by_question_type(run_result.sample_results)
    display_keys = _pick_display_keys(run_result.sample_results)

    if not display_keys:
        console.print("[yellow]No retrieval metrics to display.[/yellow]")
        return

    table = Table(title="Metrics by Question Type", show_lines=True)
    table.add_column("Question Type", style="bold", min_width=18, max_width=30)
    table.add_column("Count", justify="right", min_width=5, max_width=8)
    for key in display_keys:
        table.add_column(key, justify="right", min_width=10, max_width=14)

    for qtype in sorted(groups.keys()):
        samples = groups[qtype]
        row: list[str] = [qtype, str(len(samples))]
        for key in display_keys:
            if key == "qa_accuracy":
                qa_acc = _qa_accuracy_for(samples)
                row.append(f"{qa_acc:.4f}" if qa_acc is not None else "-")
            else:
                vals = [s.retrieval_metrics.get(key, 0.0) for s in samples]
                row.append(f"{_mean(vals):.4f}")
        table.add_row(*row)

    # Overall row
    all_samples = run_result.sample_results
    overall_row: list[str] = ["[bold]Overall[/bold]", str(len(all_samples))]
    for key in display_keys:
        if key == "qa_accuracy":
            qa_acc = _qa_accuracy_for(all_samples)
            overall_row.append(f"[bold]{qa_acc:.4f}[/bold]" if qa_acc is not None else "-")
        else:
            vals = [s.retrieval_metrics.get(key, 0.0) for s in all_samples]
            overall_row.append(f"[bold]{_mean(vals):.4f}[/bold]")
    table.add_row(*overall_row)

    console.print(table)

    # -- Timing summary -------------------------------------------------------
    agg = run_result.aggregate_metrics
    timing_keys = [k for k in sorted(agg.keys()) if k.endswith("_seconds")]
    if timing_keys:
        console.print()
        timing_table = Table(title="Timing Summary", show_lines=True)
        timing_table.add_column("Metric", style="bold", min_width=20)
        timing_table.add_column("Value (s)", justify="right", min_width=10)
        for key in timing_keys:
            timing_table.add_row(key, f"{agg[key]:.4f}")
        console.print(timing_table)

    console.print()


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    """Print a side-by-side comparison table from multiple summary.json dicts.

    Each summary dict is expected to have at minimum:
    ``adapter_name``, ``benchmark_name``, ``split``, ``aggregate_metrics``.

    Skips timing metrics to focus on quality metrics. Highlights best value
    per metric in bold green and shows percentage difference from best.
    """
    console = Console()
    console.print()

    if not summaries:
        console.print("[yellow]No results to compare.[/yellow]")
        return

    # Collect quality metric keys (skip timing).
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(s.get("aggregate_metrics", {}).keys())
    metric_keys = sorted(k for k in all_keys if not k.endswith("_seconds"))

    # Build column labels from adapter names.
    labels = [f"{s.get('adapter_name', '?')} ({s.get('split', '?')})" for s in summaries]

    table = Table(title="Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    for label in labels:
        table.add_column(label, justify="right")

    for key in metric_keys:
        row: list[str] = [key]
        values = [s.get("aggregate_metrics", {}).get(key) for s in summaries]
        numeric = [v for v in values if v is not None]
        best: float | None = None
        if numeric:
            best = max(numeric)

        for v in values:
            if v is None:
                row.append("-")
            elif best is not None and v == best:
                row.append(f"[bold green]{v:.4f}[/bold green]")
            elif best is not None and best != 0:
                pct_diff = ((v - best) / abs(best)) * 100
                row.append(f"{v:.4f} ({pct_diff:+.1f}%)")
            else:
                row.append(f"{v:.4f}")
        table.add_row(*row)

    console.print(table)
    console.print()


def format_comparison_markdown(summaries: list[dict[str, Any]]) -> str:
    """Format a side-by-side comparison as a Markdown table.

    Skips timing metrics. Highlights best value per metric in bold and
    shows percentage difference from best.
    """
    if not summaries:
        return "*No results to compare.*\n"

    # Collect quality metric keys (skip timing).
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(s.get("aggregate_metrics", {}).keys())
    metric_keys = sorted(k for k in all_keys if not k.endswith("_seconds"))

    labels = [f"{s.get('adapter_name', '?')} ({s.get('split', '?')})" for s in summaries]

    lines: list[str] = []
    # Header
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join(["---:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)

    for key in metric_keys:
        values = [s.get("aggregate_metrics", {}).get(key) for s in summaries]
        numeric = [v for v in values if v is not None]
        best: float | None = None
        if numeric:
            best = max(numeric)

        cells: list[str] = [key]
        for v in values:
            if v is None:
                cells.append("-")
            elif best is not None and v == best:
                cells.append(f"**{v:.4f}**")
            elif best is not None and best != 0:
                pct_diff = ((v - best) / abs(best)) * 100
                cells.append(f"{v:.4f} ({pct_diff:+.1f}%)")
            else:
                cells.append(f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)
