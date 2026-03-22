"""``mem-bench download`` command."""

from __future__ import annotations

import sys

import click

from mem_bench.cli._benchmarks import get_benchmark

# Map benchmark names to their known split names.
# This avoids hardcoding imports and works for all registered benchmarks.
_BENCHMARK_SPLITS: dict[str, list[str]] = {
    "longmemeval": ["oracle", "s", "m"],
    "halumem": ["medium", "long"],
    "locomo": ["test"],
}


@click.command()
@click.argument("benchmark_name")
@click.option(
    "--split",
    "-s",
    default=None,
    help="Specific split to download (default: all available splits).",
)
def download(benchmark_name: str, split: str | None) -> None:
    """Pre-download benchmark data for offline use."""
    try:
        bench = get_benchmark(benchmark_name)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Determine which splits to download.
    splits_to_download: list[str]
    if split is not None:
        splits_to_download = [split]
    else:
        # Look up known splits for this benchmark, or try to discover them
        # from the benchmark module's _SPLIT_FILES attribute.
        if benchmark_name in _BENCHMARK_SPLITS:
            splits_to_download = _BENCHMARK_SPLITS[benchmark_name]
        else:
            # Fallback: try to find _SPLIT_FILES on the benchmark class.
            split_files = getattr(type(bench), "_SPLIT_FILES", None)
            if split_files and isinstance(split_files, dict):
                splits_to_download = list(split_files.keys())
            else:
                splits_to_download = ["oracle"]

    for s in splits_to_download:
        click.echo(f"Downloading {benchmark_name} split={s} ...")
        try:
            bench.load(split=s, limit=1)
            click.echo(f"  Done: {s}")
        except Exception as exc:
            click.echo(f"  Failed ({s}): {exc}", err=True)

    click.echo("Download complete.")
