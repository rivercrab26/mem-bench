"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AdapterConfig(BaseModel):
    name: str = "bm25"
    options: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class MetricsConfig(BaseModel):
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    include_latency: bool = True
    include_cost: bool = False
    compute_semantic: bool = False
    semantic_retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    semantic_judge_model: str = "claude-haiku-4-5-20251001"


class ReportingConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["console", "json"])


class RunConfig(BaseModel):
    benchmark: str = "longmemeval"
    split: str = "oracle"
    limit: int = 0  # 0 = all
    output_dir: str = "./mem_bench_results"
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def load_config(path: str | Path | None = None) -> RunConfig:
    """Load config from TOML file, falling back to defaults."""
    if path is None:
        return RunConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Flatten [run] section to top level
    run_data = data.get("run", {})
    run_data["adapter"] = data.get("adapter", {})
    run_data["judge"] = data.get("judge", {})
    run_data["metrics"] = data.get("metrics", {})
    run_data["reporting"] = data.get("reporting", {})

    return RunConfig(**run_data)
