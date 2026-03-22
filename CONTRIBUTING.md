# Contributing to mem-bench

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/rivercrab26/mem-bench.git
cd mem-bench
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Adding a New Adapter

1. Create `src/mem_bench/adapters/your_adapter.py`
2. Implement 3 methods: `ingest()`, `recall()`, `cleanup()`
3. Register in `src/mem_bench/adapters/__init__.py`
4. Add entry point in `pyproject.toml`
5. Add optional dependency group in `pyproject.toml`
6. Write a test in `tests/adapters/`

See `examples/custom_adapter.py` for a minimal example.

## Adding a New Benchmark

1. Create `src/mem_bench/benchmarks/your_benchmark.py`
2. Implement the `Benchmark` protocol: `load()`, `__iter__()`, `__len__()`
3. Register in `src/mem_bench/cli/_benchmarks.py`
4. Add entry point in `pyproject.toml`

See `src/mem_bench/benchmarks/longmemeval.py` for a complete reference.

## Pull Request Guidelines

- One feature per PR
- Include tests for new functionality
- Run `ruff check` and `ruff format` before submitting
- Update CHANGELOG.md for user-facing changes
