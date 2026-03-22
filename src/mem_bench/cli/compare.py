"""``mem-bench compare`` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mem_bench.reporting.console import format_comparison_markdown, print_comparison


@click.command()
@click.argument("directories", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["console", "markdown"], case_sensitive=False),
    default="console",
    help="Output format: console (rich table) or markdown.",
)
def compare(directories: tuple[str, ...], output_format: str) -> None:
    """Compare results from multiple benchmark runs.

    Pass two or more result directories containing summary.json files.
    """
    if len(directories) < 2:
        click.echo("Error: at least two result directories are required.", err=True)
        sys.exit(1)

    summaries: list[dict] = []
    for dir_path in directories:
        summary_file = Path(dir_path) / "summary.json"
        if not summary_file.exists():
            click.echo(f"Error: {summary_file} not found.", err=True)
            sys.exit(1)
        with open(summary_file, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if output_format == "markdown":
        click.echo(format_comparison_markdown(summaries))
    else:
        print_comparison(summaries)
