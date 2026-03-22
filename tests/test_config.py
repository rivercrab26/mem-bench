"""Tests for configuration loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mem_bench.core.config import RunConfig, load_config


class TestDefaultRunConfig:
    """Test RunConfig defaults."""

    def test_default_benchmark(self):
        config = RunConfig()
        assert config.benchmark == "longmemeval"

    def test_default_split(self):
        config = RunConfig()
        assert config.split == "oracle"

    def test_default_limit(self):
        config = RunConfig()
        assert config.limit == 0

    def test_default_adapter(self):
        config = RunConfig()
        assert config.adapter.name == "bm25"

    def test_default_judge_disabled(self):
        config = RunConfig()
        assert config.judge.enabled is False

    def test_default_retrieval_k(self):
        config = RunConfig()
        assert config.metrics.retrieval_k == [1, 3, 5, 10]

    def test_default_reporting_formats(self):
        config = RunConfig()
        assert "console" in config.reporting.formats
        assert "json" in config.reporting.formats

    def test_model_dump_roundtrip(self):
        config = RunConfig()
        data = config.model_dump()
        restored = RunConfig(**data)
        assert restored == config


class TestLoadConfig:
    """Test load_config from TOML file."""

    def test_load_none_returns_defaults(self):
        config = load_config(None)
        assert config == RunConfig()

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.toml")

    def test_load_from_toml(self, tmp_path):
        toml_content = """\
[run]
benchmark = "locomo"
split = "test"
limit = 50

[adapter]
name = "mem0"

[judge]
enabled = true
model = "gpt-4o"
provider = "openai"

[metrics]
retrieval_k = [1, 5, 10]
"""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_content)

        config = load_config(toml_path)

        assert config.benchmark == "locomo"
        assert config.split == "test"
        assert config.limit == 50
        assert config.adapter.name == "mem0"
        assert config.judge.enabled is True
        assert config.judge.model == "gpt-4o"
        assert config.metrics.retrieval_k == [1, 5, 10]

    def test_load_partial_toml(self, tmp_path):
        """A TOML with only some fields should use defaults for the rest."""
        toml_content = """\
[run]
benchmark = "halumem"
"""
        toml_path = tmp_path / "partial.toml"
        toml_path.write_text(toml_content)

        config = load_config(toml_path)

        assert config.benchmark == "halumem"
        assert config.split == "oracle"  # default
        assert config.adapter.name == "bm25"  # default
