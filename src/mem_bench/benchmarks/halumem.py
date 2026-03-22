"""HaluMem benchmark (2025) - Memory Hallucination Evaluation.

Loads the HaluMem dataset (20 users with multi-session dialogues) and yields
``BenchmarkSample`` objects for the memory QA task.

Two variants are available:
- **medium**: ~160k tokens avg context per user, ~70 sessions/user
- **long**: ~1M tokens avg context per user, ~120 sessions/user

Data source: https://github.com/MemTensor/HaluMem
HuggingFace: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from mem_bench.benchmarks._download import download_benchmark
from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem

logger = logging.getLogger(__name__)

HF_REPO_ID = "IAAR-Shanghai/HaluMem"

# Mapping from user-facing split names to filenames in the HF repo.
_SPLIT_FILES: dict[str, str] = {
    "medium": "HaluMem-Medium.jsonl",
    "long": "HaluMem-Long.jsonl",
}


def _flatten_dialogue(dialogue: list[dict[str, Any]]) -> str:
    """Flatten a list of dialogue turns into a single text block."""
    lines: list[str] = []
    for turn in dialogue:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_halumem_timestamp(ts: str | None) -> str | None:
    """Best-effort parse of HaluMem timestamps to ISO 8601.

    HaluMem uses formats like ``'Dec 15, 2025, 06:11:23'``.
    Returns ``None`` for unparseable values.
    """
    if not ts:
        return None
    try:
        from datetime import datetime

        dt = datetime.strptime(ts, "%b %d, %Y, %H:%M:%S")
        return dt.isoformat()
    except (ValueError, TypeError):
        pass
    return None


class HaluMemBenchmark:
    """Loader for the HaluMem benchmark.

    Implements the ``Benchmark`` protocol defined in
    ``mem_bench.core.benchmark``.
    """

    def __init__(self, *, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = cache_dir
        self._samples: list[BenchmarkSample] = []

    # -- Protocol properties --------------------------------------------------

    @property
    def name(self) -> str:
        return "halumem"

    @property
    def version(self) -> str:
        return "1.0"

    # -- Protocol methods -----------------------------------------------------

    def load(self, *, split: str = "medium", limit: int | None = None) -> None:
        """Download (if needed) and load the dataset into memory.

        Args:
            split: One of ``"medium"`` or ``"long"``.
            limit: Maximum number of QA samples to load.
                ``None`` or ``0`` means all.
        """
        if split not in _SPLIT_FILES:
            raise ValueError(f"Unknown split {split!r}. Choose from: {list(_SPLIT_FILES)}")

        filename = _SPLIT_FILES[split]
        path = download_benchmark(HF_REPO_ID, filename, cache_dir=self._cache_dir)

        logger.info("Loading HaluMem split=%s from %s", split, path)

        # HaluMem files are JSONL (one JSON object per line, one per user).
        users: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    users.append(json.loads(line))

        samples: list[BenchmarkSample] = []
        for user_data in users:
            new_samples = self._convert_user(user_data)
            samples.extend(new_samples)

        if limit and limit > 0:
            samples = samples[:limit]

        self._samples = samples
        logger.info("Loaded %d QA samples from %d users", len(self._samples), len(users))

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` objects for the loaded data."""
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _convert_user(
        user_data: dict[str, Any],
    ) -> list[BenchmarkSample]:
        """Convert a single HaluMem user record into BenchmarkSample objects.

        Each QA pair from all sessions becomes one BenchmarkSample. All QA
        pairs for a user share the same set of IngestItems (the dialogue
        sessions).
        """
        uuid = user_data.get("uuid", "unknown")
        persona_info = user_data.get("persona_info", {})
        sessions = user_data.get("sessions", [])

        # Build IngestItems from sessions.
        ingest_items: list[IngestItem] = []
        # Collect all questions across sessions.
        all_questions: list[tuple[int, int, dict[str, Any]]] = []

        for sess_idx, session in enumerate(sessions):
            dialogue = session.get("dialogue", [])
            start_time = session.get("start_time")
            end_time = session.get("end_time")
            memory_points = session.get("memory_points", [])

            text = _flatten_dialogue(dialogue)
            iso_ts = _parse_halumem_timestamp(start_time)
            doc_id = f"{uuid}_session_{sess_idx}"

            # Collect memory point metadata for this session.
            mp_summary: list[dict[str, Any]] = []
            for mp in memory_points:
                mp_summary.append(
                    {
                        "index": mp.get("index"),
                        "memory_type": mp.get("memory_type", ""),
                        "memory_source": mp.get("memory_source", ""),
                        "is_update": mp.get("is_update", False),
                    }
                )

            ingest_items.append(
                IngestItem(
                    content=text,
                    document_id=doc_id,
                    metadata={
                        "user_uuid": uuid,
                        "session_index": sess_idx,
                        "start_time": start_time or "",
                        "end_time": end_time or "",
                        "dialogue_turn_num": session.get("dialogue_turn_num", 0),
                        "dialogue_token_length": session.get("dialogue_token_length", 0),
                        "memory_point_count": len(memory_points),
                        "memory_points_summary": mp_summary,
                    },
                    timestamp=iso_ts,
                )
            )

            # Collect questions from this session.
            questions = session.get("questions", [])
            for q_idx, q in enumerate(questions):
                all_questions.append((sess_idx, q_idx, q))

        # Build BenchmarkSample for each QA pair.
        results: list[BenchmarkSample] = []
        for sess_idx, q_idx, qa in all_questions:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            evidence = qa.get("evidence", [])
            difficulty = qa.get("difficulty", "unknown")
            question_type = qa.get("question_type", "unknown")

            # Resolve evidence to ground-truth doc IDs.
            # Evidence items contain memory_content and memory_type; we link
            # back to the session that sourced the question.
            ground_truth_doc_ids = [f"{uuid}_session_{sess_idx}"]

            sid = f"{uuid}_s{sess_idx}_q{q_idx}"

            # Build evidence metadata.
            evidence_meta: list[dict[str, str]] = []
            for ev in evidence:
                evidence_meta.append(
                    {
                        "memory_content": ev.get("memory_content", ""),
                        "memory_type": ev.get("memory_type", ""),
                    }
                )

            results.append(
                BenchmarkSample(
                    sample_id=sid,
                    question=question,
                    reference_answer=str(answer),
                    question_type=question_type,
                    ingest_items=ingest_items,
                    ground_truth_doc_ids=ground_truth_doc_ids,
                    metadata={
                        "user_uuid": uuid,
                        "session_index": sess_idx,
                        "difficulty": difficulty,
                        "evidence": evidence_meta,
                        "persona_info": persona_info,
                    },
                )
            )

        return results
