# mem-bench

Standardized benchmark framework for AI memory systems.

## Quick Start

```bash
pip install mem-bench

# Run BM25 baseline on LongMemEval
mem-bench run --adapter bm25 --benchmark longmemeval --split oracle --limit 10

# Compare multiple systems
mem-bench compare results/bm25/ results/mem0/ --format markdown
```

## Supported Adapters

| Adapter | Memory System | Type |
|---------|--------------|------|
| `bm25` | BM25 (baseline) | Sparse retrieval |
| `mem0` | Mem0 | Vector + fact extraction |
| `graphiti` | Graphiti/Zep | Temporal knowledge graph |
| `langmem` | LangChain/LangMem | LangGraph memory |
| `letta` | Letta (MemGPT) | Tiered memory |
| `hindsight` | OpenClaw/Hindsight | Biomimetic memory |

## Supported Benchmarks

- **LongMemEval** (ICLR 2025) - 5 core memory abilities, 500 questions
- **LoCoMo** (ACL 2024) - Long-term conversational memory
- **HaluMem** (2025) - Memory hallucination diagnosis

## Writing a Custom Adapter

```python
from mem_bench import IngestItem, RecallQuery, RecallResult

class MyAdapter:
    def ingest(self, items, *, namespace="default"):
        ...
    def recall(self, query, *, namespace="default"):
        ...
    def cleanup(self, *, namespace="default"):
        ...
```

```bash
mem-bench run --adapter mypackage:MyAdapter --benchmark longmemeval
```

## License

Apache 2.0
