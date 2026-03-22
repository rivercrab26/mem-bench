"""``mem-bench list`` commands."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from mem_bench.adapters import list_adapters
from mem_bench.cli._benchmarks import list_benchmarks


@click.group("list")
def list_group() -> None:
    """List available adapters or benchmarks."""


@list_group.command("adapters")
def list_adapters_cmd() -> None:
    """List all registered memory adapters."""
    console = Console()
    names = list_adapters()

    table = Table(title="Available Adapters")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="bold cyan")

    for idx, name in enumerate(names, start=1):
        table.add_row(str(idx), name)

    console.print(table)


@list_group.command("benchmarks")
def list_benchmarks_cmd() -> None:
    """List all registered benchmarks."""
    console = Console()
    names = list_benchmarks()

    table = Table(title="Available Benchmarks")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="bold cyan")

    for idx, name in enumerate(names, start=1):
        table.add_row(str(idx), name)

    console.print(table)
