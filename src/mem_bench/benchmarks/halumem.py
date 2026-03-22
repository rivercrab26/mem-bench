"""HaluMem benchmark (2025) - Memory Hallucination Diagnosis.

TODO: Implement full HaluMem support.
Dataset: https://github.com/MemTensor/HaluMem
"""

from __future__ import annotations

from typing import Iterator

from mem_bench.core.benchmark import BenchmarkSample


class HaluMemBenchmark:
    """HaluMem benchmark loader. Not yet implemented."""

    @property
    def name(self) -> str:
        return "halumem"

    @property
    def version(self) -> str:
        return "1.0"

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        raise NotImplementedError(
            "HaluMem benchmark is not yet implemented. "
            "Contributions welcome: https://github.com/rivercrab26/mem-bench"
        )

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter([])

    def __len__(self) -> int:
        return 0
