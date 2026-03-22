<p align="center">
  <h1 align="center">mem-bench</h1>
  <p align="center">Standardized benchmark framework for AI memory systems</p>
</p>

<p align="center">
  <a href="https://github.com/rivercrab26/mem-bench/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://github.com/rivercrab26/mem-bench"><img src="https://img.shields.io/github/stars/rivercrab26/mem-bench?style=social" alt="GitHub Stars"></a>
</p>

---

**mem-bench** is a neutral, pluggable benchmark tool for AI memory systems. Any memory system can be tested against established academic benchmarks by implementing just 3 methods.

## Why mem-bench?

Every memory system reports scores on different benchmarks using different methodologies. **There's no way to compare them fairly.** mem-bench fixes this by providing:

- **Unified adapter interface** -- 3 methods: `ingest()`, `recall()`, `cleanup()`
- **Academic benchmarks** -- LongMemEval (ICLR 2025), LoCoMo (ACL 2024), HaluMem
- **One command** -- `mem-bench run --adapter mem0 --benchmark longmemeval`
- **Side-by-side comparison** -- `mem-bench compare results/mem0/ results/graphiti/`

## Quick Start

```bash
pip install mem-bench

# Run BM25 baseline on LongMemEval (no API keys needed)
mem-bench run --adapter bm25 --benchmark longmemeval --split oracle --limit 10

# Test Mem0
pip install mem-bench[mem0]
mem-bench run --adapter mem0 --benchmark longmemeval --split oracle

# Compare systems
mem-bench compare results/bm25_longmemeval/ results/mem0_longmemeval/ --format markdown
```

## Supported Adapters

| Adapter | Memory System | Stars | Type | Install |
|---------|--------------|-------|------|---------|
| `bm25` | BM25 (baseline) | -- | Sparse retrieval | Built-in |
| `mem0` | [Mem0](https://github.com/mem0ai/mem0) | 50k+ | Vector + fact extraction | `pip install mem-bench[mem0]` |
| `graphiti` | [Graphiti/Zep](https://github.com/getzep/graphiti) | 24k+ | Temporal knowledge graph | `pip install mem-bench[graphiti]` |
| `langmem` | [LangChain/LangMem](https://github.com/langchain-ai/langmem) | -- | LangGraph memory | `pip install mem-bench[langchain]` |
| `letta` | [Letta (MemGPT)](https://github.com/letta-ai/letta) | 21k+ | Tiered memory | `pip install mem-bench[letta]` |
| `hindsight` | [OpenClaw/Hindsight](https://github.com/openclaw) | -- | Biomimetic memory | Built-in (HTTP) |

## Supported Benchmarks

| Benchmark | Venue | Questions | What it tests |
|-----------|-------|-----------|---------------|
| **LongMemEval** | ICLR 2025 | 500 | Information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention |
| **LoCoMo** | ACL 2024 | 81 | Single-hop, multi-hop, temporal, open-domain, adversarial reasoning |
| **HaluMem** | 2025 | 3,500+ | Memory extraction accuracy, update accuracy, QA hallucination |

## CLI Commands

```bash
# Run a benchmark
mem-bench run --adapter <name> --benchmark <benchmark> [options]
  --split <split>          # oracle, s, m (benchmark-specific)
  --limit <N>              # max samples (for quick testing)
  --output-dir <path>      # where to save results
  --config <path>          # TOML config file
  --no-judge               # skip LLM-as-Judge, retrieval metrics only

# List available components
mem-bench list adapters
mem-bench list benchmarks

# Compare multiple runs
mem-bench compare <dir1> <dir2> ... --format markdown

# Pre-download benchmark data
mem-bench download longmemeval --split oracle
```

## Writing a Custom Adapter

Just implement 3 methods. No base class required (uses Python Protocol):

```python
class MyMemorySystem:
    def ingest(self, items, *, namespace="default"):
        """Store items into your memory system."""
        for item in items:
            self.store.add(item.content, id=item.document_id)

    def recall(self, query, *, namespace="default"):
        """Search and return ranked results."""
        hits = self.store.search(query.query, limit=query.top_k)
        return [RecallResult(document_id=h.id, content=h.text, score=h.score) for h in hits]

    def cleanup(self, *, namespace="default"):
        """Delete all data in this namespace."""
        self.store.delete_all(namespace)
```

```bash
mem-bench run --adapter mypackage:MyMemorySystem --benchmark longmemeval
```

Or register via entry points in `pyproject.toml`:

```toml
[project.entry-points."mem_bench.adapters"]
my-memory = "mypackage.adapter:MyMemorySystem"
```

## Configuration

```toml
# mem-bench.toml
[run]
benchmark = "longmemeval"
split = "s"
limit = 50

[adapter]
name = "mem0"
[adapter.options]
api_key_env = "MEM0_API_KEY"

[judge]
enabled = true
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

[metrics]
retrieval_k = [1, 3, 5, 10]
include_latency = true
```

```bash
mem-bench run --config mem-bench.toml
```

## Output

Each run produces a results directory:

```
mem_bench_results/
  results.jsonl        # Per-sample detailed results
  summary.json         # Aggregated metrics
  report.md            # Human-readable report
```

## Metrics

**Retrieval** (always computed, no API keys needed):
- `recall_any@k` -- Did any correct document appear in top-k?
- `recall_all@k` -- Did all correct documents appear in top-k?
- `ndcg@k` -- Normalized discounted cumulative gain
- `mrr` -- Mean reciprocal rank

**QA Accuracy** (requires `--judge-model` or config, uses LLM-as-Judge):
- Per-question-type accuracy
- Task-averaged accuracy
- Abstention accuracy

**Latency**:
- Ingest and recall times (mean, p95, p99)

## Contributing

Contributions welcome! Key areas:

- **New adapters** -- Add support for more memory systems
- **New benchmarks** -- Integrate additional academic benchmarks
- **Improved evaluation** -- Better metrics and reporting

See [examples/custom_adapter.py](examples/custom_adapter.py) for adapter examples.

## License

Apache 2.0
