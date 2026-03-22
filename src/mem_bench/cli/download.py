"""``mem-bench download`` command."""

from __future__ import annotations

import sys

import click

from mem_bench.cli._benchmarks import get_benchmark


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
    # Benchmarks that expose a _SPLIT_FILES mapping allow downloading
    # individual splits; otherwise we download the default split.
    splits_to_download: list[str]
    if split is not None:
        splits_to_download = [split]
    else:
        # Try to get all known splits from the benchmark module.
        from mem_bench.benchmarks import longmemeval as _lme

        _split_map: dict[str, str] | None = getattr(_lme, "_SPLIT_FILES", None)
        if benchmark_name == "longmemeval" and _split_map is not None:
            splits_to_download = list(_split_map.keys())
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
