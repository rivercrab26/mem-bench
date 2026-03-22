"""Rich console output for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from mem_bench.core.runner import RunResult
from mem_bench.core.types import SampleResult


# Key metrics to display (avoids truncation from too many columns).
_DISPLAY_METRICS = [
    "recall_any@1",
    "recall_all@5",
    "ndcg@5",
    "mrr",
    "qa_accuracy",
]


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


def _qa_accuracy_for(samples: list[SampleResult]) -> float | None:
    """Compute QA accuracy for a list of samples, or None if no QA scores."""
    scores = [s.qa_score for s in samples if s.qa_score is not None]
    if not scores:
        return None
    return sum(1 for s in scores if s > 0.5) / len(scores)


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
