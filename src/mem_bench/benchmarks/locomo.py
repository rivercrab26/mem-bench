"""LoCoMo benchmark (ACL 2024) - Long-term Conversational Memory.

TODO: Implement full LoCoMo support.
Dataset: https://github.com/snap-research/locomo
"""

from __future__ import annotations

from typing import Iterator

from mem_bench.core.benchmark import BenchmarkSample


class LoCoMoBenchmark:
    """LoCoMo benchmark loader. Not yet implemented."""

    @property
    def name(self) -> str:
        return "locomo"

    @property
    def version(self) -> str:
        return "1.0"

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        raise NotImplementedError(
            "LoCoMo benchmark is not yet implemented. "
            "Contributions welcome: https://github.com/rivercrab26/mem-bench"
        )

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter([])

    def __len__(self) -> int:
        return 0
