"""``mem-bench run`` command."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from mem_bench.adapters import get_adapter
from mem_bench.cli._benchmarks import get_benchmark
from mem_bench.core.config import AdapterConfig, load_config
from mem_bench.core.runner import BenchmarkRunner
from mem_bench.reporting.console import print_results
from mem_bench.reporting.json_report import save_json_report
from mem_bench.reporting.markdown_report import save_markdown_report

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--adapter",
    "-a",
    default=None,
    help="Memory adapter name (e.g. bm25, mem0, graphiti).",
)
@click.option(
    "--benchmark",
    "-b",
    default=None,
    help="Benchmark name (e.g. longmemeval).",
)
@click.option(
    "--split",
    "-s",
    default=None,
    help="Dataset split to use (e.g. oracle, s, m).",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Max number of samples to evaluate (0 = all).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to TOML config file.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Directory for saving results.",
)
@click.option(
    "--no-judge",
    is_flag=True,
    default=False,
    help="Disable LLM judge evaluation.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose/debug logging.",
)
def run(
    adapter: str | None,
    benchmark: str | None,
    split: str | None,
    limit: int | None,
    config_path: str | None,
    output_dir: str | None,
    no_judge: bool,
    verbose: bool,
) -> None:
    """Run a benchmark against a memory adapter."""
    # -- Logging setup --------------------------------------------------------
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Config: file first, then CLI overrides --------------------------------
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if adapter is not None:
        cfg.adapter = AdapterConfig(name=adapter)
    if benchmark is not None:
        cfg.benchmark = benchmark
    if split is not None:
        cfg.split = split
    if limit is not None:
        cfg.limit = limit
    if output_dir is not None:
        cfg.output_dir = output_dir
    if no_judge:
        cfg.judge.enabled = False

    click.echo(f"Adapter:   {cfg.adapter.name}")
    click.echo(f"Benchmark: {cfg.benchmark}")
    click.echo(f"Split:     {cfg.split}")
    click.echo(f"Limit:     {cfg.limit or 'all'}")
    click.echo()

    # -- Adapter ---------------------------------------------------------------
    try:
        mem_adapter = get_adapter(cfg.adapter.name, cfg.adapter.options)
    except (ValueError, ImportError) as exc:
        click.echo(f"Error loading adapter: {exc}", err=True)
        sys.exit(1)

    # -- Benchmark -------------------------------------------------------------
    try:
        bench = get_benchmark(cfg.benchmark)
    except ValueError as exc:
        click.echo(f"Error loading benchmark: {exc}", err=True)
        sys.exit(1)

    bench.load(split=cfg.split, limit=cfg.limit or None)
    click.echo(f"Loaded {len(bench)} samples.")

    # -- Run -------------------------------------------------------------------
    runner = BenchmarkRunner(mem_adapter, bench, cfg)
    result = runner.run()

    # -- Output ----------------------------------------------------------------
    formats = {f.lower() for f in cfg.reporting.formats}

    if "console" in formats:
        print_results(result)

    out_dir = Path(cfg.output_dir)
    if "json" in formats:
        save_json_report(result, out_dir)
    if "markdown" in formats:
        save_markdown_report(result, out_dir)
    if "html" in formats:
        from mem_bench.reporting.html_report import save_html_report

        save_html_report(result, out_dir)

    click.echo()
    click.echo(
        f"Completed in {result.total_seconds:.1f}s  "
        f"({result.num_samples} samples, {result.num_failed} failed)"
    )
    if out_dir.exists():
        click.echo(f"Results saved to {out_dir}")
