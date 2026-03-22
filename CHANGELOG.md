# Changelog

## 0.1.0 (2026-03-22)

Initial release.

### Features
- Core Protocol-based adapter interface (ingest/recall/cleanup)
- 6 adapters: BM25, Mem0, Graphiti, LangMem, Letta, Hindsight
- 3 benchmarks: LongMemEval (full), LoCoMo, HaluMem
- Retrieval metrics: recall@k, nDCG@k, MRR
- QA evaluation: LLM-as-Judge with OpenAI and Anthropic support
- CLI: run, list, compare, download commands
- Reporting: console (Rich), JSON, Markdown
- Runner with warmup, retry, and per-sample namespace isolation
- GitHub Actions CI with lint, test, typecheck
