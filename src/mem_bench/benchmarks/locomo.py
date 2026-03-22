"""LoCoMo benchmark (ACL 2024) - Long-term Conversational Memory.

Loads the LoCoMo dataset (10 long conversations with ~300 turns each) and
yields ``BenchmarkSample`` objects for the question-answering task.

Data source: https://github.com/snap-research/locomo
HuggingFace mirror: https://huggingface.co/datasets/Percena/locomo-mc10
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

# The Percena/locomo-mc10 HF repo contains the original locomo10.json in raw/.
HF_REPO_ID = "Percena/locomo-mc10"
_FILENAME = "raw/locomo10.json"

# QA category integer -> human-readable label.
_CATEGORY_LABELS: dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def _extract_sessions(
    conversation: dict[str, Any],
) -> list[tuple[str, str | None, list[dict[str, Any]]]]:
    """Extract ordered sessions from a LoCoMo conversation dict.

    Returns a list of (session_key, date_time, turns) tuples sorted by
    session number.
    """
    # Find all session keys (session_1, session_2, ...).
    session_keys: list[tuple[int, str]] = []
    for key in conversation:
        m = re.match(r"^session_(\d+)$", key)
        if m:
            session_keys.append((int(m.group(1)), key))

    session_keys.sort()

    results: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    for num, key in session_keys:
        date_key = f"session_{num}_date_time"
        date_time = conversation.get(date_key)
        turns = conversation[key]
        results.append((key, date_time, turns))

    return results


def _flatten_turns(turns: list[dict[str, Any]]) -> str:
    """Flatten a list of LoCoMo dialogue turns into a single text block."""
    lines: list[str] = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _parse_locomo_date(date_str: str | None) -> str | None:
    """Best-effort parse of LoCoMo date strings to ISO 8601.

    Handles formats like:
    - ``'1:56 pm on 8 May, 2023'``
    - ``'10:30 am on 15 January, 2024'``

    Returns ``None`` for unparseable values.
    """
    if not date_str:
        return None
    try:
        from datetime import datetime

        # Try common LoCoMo format: "1:56 pm on 8 May, 2023"
        m = re.match(
            r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
            date_str,
            re.IGNORECASE,
        )
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
            ampm = m.group(3).lower()
            day = int(m.group(4))
            month_str = m.group(5)
            year = int(m.group(6))

            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

            month_map = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }
            month = month_map.get(month_str.lower())
            if month:
                dt = datetime(year, month, day, hour, minute)
                return dt.isoformat()
    except Exception:
        pass
    return None


class LoCoMoBenchmark:
    """Loader for the LoCoMo benchmark.

    Implements the ``Benchmark`` protocol defined in
    ``mem_bench.core.benchmark``.
    """

    def __init__(self, *, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = cache_dir
        self._samples: list[BenchmarkSample] = []

    # -- Protocol properties --------------------------------------------------

    @property
    def name(self) -> str:
        return "locomo"

    @property
    def version(self) -> str:
        return "1.0"

    # -- Protocol methods -----------------------------------------------------

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        """Download (if needed) and load the dataset into memory.

        Args:
            split: Only ``"test"`` is supported (all 10 conversations).
            limit: Maximum number of QA samples to load.
                ``None`` or ``0`` means all.
        """
        if split != "test":
            raise ValueError(f"Unknown split {split!r}. LoCoMo only supports 'test'.")

        path = download_benchmark(HF_REPO_ID, _FILENAME, cache_dir=self._cache_dir)

        logger.info("Loading LoCoMo from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = json.load(f)

        samples: list[BenchmarkSample] = []
        for conv_item in raw_data:
            new_samples = self._convert_conversation(conv_item)
            samples.extend(new_samples)

        if limit and limit > 0:
            samples = samples[:limit]

        self._samples = samples
        logger.info("Loaded %d QA samples from %d conversations", len(self._samples), len(raw_data))

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` objects for the loaded data."""
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _convert_conversation(
        conv_item: dict[str, Any],
    ) -> list[BenchmarkSample]:
        """Convert a single LoCoMo conversation into BenchmarkSample objects.

        Each QA pair becomes one BenchmarkSample. All QA pairs from the same
        conversation share the same set of IngestItems (the conversation
        sessions).
        """
        conversation = conv_item.get("conversation", {})
        qa_list = conv_item.get("qa", [])
        sample_id_prefix = conv_item.get("sample_id", "locomo_unknown")

        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")

        # Build IngestItems from sessions.
        sessions = _extract_sessions(conversation)
        ingest_items: list[IngestItem] = []
        doc_id_map: dict[str, str] = {}  # dia_id prefix -> document_id

        for session_key, date_time, turns in sessions:
            text = _flatten_turns(turns)
            iso_ts = _parse_locomo_date(date_time)
            doc_id = f"{sample_id_prefix}_{session_key}"

            ingest_items.append(
                IngestItem(
                    content=text,
                    document_id=doc_id,
                    metadata={
                        "session_key": session_key,
                        "date_time": date_time or "",
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                    },
                    timestamp=iso_ts,
                )
            )

            # Map dia_id prefixes to doc_id for ground truth resolution.
            # dia_ids are like "D1:0", "D1:1" for session_1.
            session_num = re.search(r"(\d+)$", session_key)
            if session_num:
                doc_id_map[f"D{session_num.group(1)}"] = doc_id

        # Build BenchmarkSample for each QA pair.
        results: list[BenchmarkSample] = []
        for qa_idx, qa in enumerate(qa_list):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            category = qa.get("category", 0)
            evidence = qa.get("evidence", [])

            question_type = _CATEGORY_LABELS.get(int(category) if category else 0, "unknown")

            # Resolve evidence dia_ids to ground truth doc IDs.
            ground_truth_doc_ids: list[str] = []
            for eid in evidence:
                # Evidence IDs look like "D1:5" -> session_1
                prefix = eid.split(":")[0] if ":" in str(eid) else str(eid)
                if prefix in doc_id_map:
                    did = doc_id_map[prefix]
                    if did not in ground_truth_doc_ids:
                        ground_truth_doc_ids.append(did)

            sid = f"{sample_id_prefix}_q{qa_idx}"

            metadata: dict[str, Any] = {
                "conversation_id": sample_id_prefix,
                "category": category,
                "evidence_ids": evidence,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            }
            if "adversarial_answer" in qa:
                metadata["adversarial_answer"] = qa["adversarial_answer"]

            results.append(
                BenchmarkSample(
                    sample_id=sid,
                    question=question,
                    reference_answer=str(answer),
                    question_type=question_type,
                    ingest_items=ingest_items,
                    ground_truth_doc_ids=ground_truth_doc_ids,
                    metadata=metadata,
                )
            )

        return results
