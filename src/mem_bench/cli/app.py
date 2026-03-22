"""mem-bench CLI application."""

from __future__ import annotations

import click

from mem_bench.cli.compare import compare
from mem_bench.cli.download import download
from mem_bench.cli.list_cmd import list_group
from mem_bench.cli.run import run


@click.group()
@click.version_option(package_name="mem-bench")
def main() -> None:
    """mem-bench: benchmark framework for AI memory systems."""


main.add_command(run)
main.add_command(list_group, name="list")
main.add_command(compare)
main.add_command(download)
