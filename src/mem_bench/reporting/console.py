"""Rich console output for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from mem_bench.core.runner import RunResult
from mem_bench.core.types import SampleResult


def _group_by_question_type(
    samples: list[SampleResult],
) -> dict[str, list[SampleResult]]:
    """Group sample results by question type."""
    groups: dict[str, list[SampleResult]] = defaultdict(list)
    for s in samples:
        groups[s.question_type].append(s)
    return dict(groups)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_keys(samples: list[SampleResult]) -> list[str]:
    """Collect all metric keys across all samples, sorted."""
    keys: set[str] = set()
    for s in samples:
        keys.update(s.retrieval_metrics.keys())
    return sorted(keys)


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
    all_keys = _metric_keys(run_result.sample_results)

    if not all_keys:
        console.print("[yellow]No retrieval metrics to display.[/yellow]")
        return

    table = Table(title="Metrics by Question Type", show_lines=True)
    table.add_column("Question Type", style="bold")
    table.add_column("Count", justify="right")
    for key in all_keys:
        table.add_column(key, justify="right")

    for qtype in sorted(groups.keys()):
        samples = groups[qtype]
        row: list[str] = [qtype, str(len(samples))]
        for key in all_keys:
            vals = [s.retrieval_metrics.get(key, 0.0) for s in samples]
            row.append(f"{_mean(vals):.4f}")
        table.add_row(*row)

    # Overall row
    all_samples = run_result.sample_results
    overall_row: list[str] = ["[bold]Overall[/bold]", str(len(all_samples))]
    for key in all_keys:
        vals = [s.retrieval_metrics.get(key, 0.0) for s in all_samples]
        overall_row.append(f"[bold]{_mean(vals):.4f}[/bold]")
    table.add_row(*overall_row)

    console.print(table)

    # -- Timing summary -------------------------------------------------------
    agg = run_result.aggregate_metrics
    timing_keys = [k for k in sorted(agg.keys()) if k.endswith("_seconds")]
    if timing_keys:
        console.print()
        timing_table = Table(title="Timing Summary")
        timing_table.add_column("Metric", style="bold")
        timing_table.add_column("Value (s)", justify="right")
        for key in timing_keys:
            timing_table.add_row(key, f"{agg[key]:.4f}")
        console.print(timing_table)

    console.print()


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    """Print a side-by-side comparison table from multiple summary.json dicts.

    Each summary dict is expected to have at minimum:
    ``adapter_name``, ``benchmark_name``, ``split``, ``aggregate_metrics``.
    """
    console = Console()
    console.print()

    if not summaries:
        console.print("[yellow]No results to compare.[/yellow]")
        return

    # Collect all metric keys across all summaries.
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(s.get("aggregate_metrics", {}).keys())
    metric_keys = sorted(all_keys)

    # Build column labels from adapter names.
    labels = [
        f"{s.get('adapter_name', '?')} ({s.get('split', '?')})"
        for s in summaries
    ]

    table = Table(title="Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    for label in labels:
        table.add_column(label, justify="right")

    for key in metric_keys:
        row: list[str] = [key]
        values = [s.get("aggregate_metrics", {}).get(key) for s in summaries]
        # Highlight the best value (highest for scores, lowest for *_seconds).
        is_timing = key.endswith("_seconds")
        numeric = [v for v in values if v is not None]
        best: float | None = None
        if numeric:
            best = min(numeric) if is_timing else max(numeric)

        for v in values:
            if v is None:
                row.append("-")
            elif v == best:
                row.append(f"[bold green]{v:.4f}[/bold green]")
            else:
                row.append(f"{v:.4f}")
        table.add_row(*row)

    console.print(table)
    console.print()
