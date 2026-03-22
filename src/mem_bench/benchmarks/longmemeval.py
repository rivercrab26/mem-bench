"""LongMemEval benchmark loader.

Loads the LongMemEval dataset (oracle / S / M splits) from HuggingFace Hub
and yields ``BenchmarkSample`` objects ready for the runner.

Data source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from mem_bench.benchmarks._download import download_benchmark
from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem

logger = logging.getLogger(__name__)

HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"

# Mapping from user-facing split names to filenames in the HF repo.
_SPLIT_FILES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def _parse_longmemeval_date(date_str: str) -> str | None:
    """Convert LongMemEval date format to ISO 8601.

    ``'2023/05/30 (Tue) 16:26'`` -> ``'2023-05-30T16:26:00'``

    Returns ``None`` for time-only strings or unparseable values.
    """
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s+\(\w+\)\s+(\d{2}):(\d{2})", date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"
    return None


def _flatten_session(turns: list[dict[str, Any]]) -> str:
    """Flatten a list of conversation turns into a single text block."""
    lines: list[str] = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class LongMemEvalBenchmark:
    """Loader for the LongMemEval benchmark.

    Implements the ``Benchmark`` protocol defined in
    ``mem_bench.core.benchmark``.
    """

    def __init__(self, *, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = cache_dir
        self._samples: list[dict[str, Any]] = []

    # -- Protocol properties --------------------------------------------------

    @property
    def name(self) -> str:
        return "longmemeval"

    @property
    def version(self) -> str:
        return "1.0"

    # -- Protocol methods -----------------------------------------------------

    def load(self, *, split: str = "oracle", limit: int | None = None) -> None:
        """Download (if needed) and load the dataset into memory.

        Args:
            split: One of ``"oracle"``, ``"s"``, ``"m"``.
            limit: Maximum number of samples to load. ``None`` or ``0`` means all.
        """
        if split not in _SPLIT_FILES:
            raise ValueError(f"Unknown split {split!r}. Choose from: {list(_SPLIT_FILES)}")

        filename = _SPLIT_FILES[split]
        path = download_benchmark(HF_REPO_ID, filename, cache_dir=self._cache_dir)

        logger.info("Loading LongMemEval split=%s from %s", split, path)
        with open(path, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)

        if limit and limit > 0:
            data = data[:limit]

        self._samples = data
        logger.info("Loaded %d samples", len(self._samples))

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` objects for the loaded data."""
        for item in self._samples:
            yield self._convert(item)

    def __len__(self) -> int:
        return len(self._samples)

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _convert(item: dict[str, Any]) -> BenchmarkSample:
        """Convert a single raw JSON entry into a ``BenchmarkSample``."""
        sessions: list[list[dict[str, Any]]] = item["haystack_sessions"]
        session_ids: list[str] = item["haystack_session_ids"]
        session_dates: list[str] = item["haystack_dates"]

        ingest_items: list[IngestItem] = []
        for sid, date_str, turns in zip(session_ids, session_dates, sessions):
            text = _flatten_session(turns)
            iso_ts = _parse_longmemeval_date(str(date_str))

            ingest_items.append(
                IngestItem(
                    content=text,
                    document_id=sid,
                    metadata={"session_id": sid, "date": str(date_str)},
                    timestamp=iso_ts,
                )
            )

        question_date = item.get("question_date", "")

        return BenchmarkSample(
            sample_id=item["question_id"],
            question=item["question"],
            reference_answer=item.get("answer", ""),
            question_type=item.get("question_type", "unknown"),
            ingest_items=ingest_items,
            ground_truth_doc_ids=item.get("answer_session_ids", []),
            metadata={
                "question_date": question_date,
                "question_date_iso": _parse_longmemeval_date(question_date),
            },
        )
