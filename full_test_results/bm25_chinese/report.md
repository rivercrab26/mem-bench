# Benchmark Report: BM25

| Field | Value |
|-------|-------|
| Benchmark | chinese |
| Split | test |
| Samples | 5 |
| Failed | 0 |
| Total Time | 9.9s |

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_hop | 5 | 0.6667 | 0.4000 | 0.9262 | 0.9262 | 0.9262 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Overall** | 5 | **0.6667** | **0.4000** | **0.9262** | **0.9262** | **0.9262** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 0.6667 |
| ndcg@1 | 0.4000 |
| ndcg@10 | 0.9262 |
| ndcg@3 | 0.9262 |
| ndcg@5 | 0.9262 |
| qa_accuracy | 1.0000 |
| recall_all@1 | 0.4000 |
| recall_all@10 | 1.0000 |
| recall_all@3 | 1.0000 |
| recall_all@5 | 1.0000 |
| recall_any@1 | 0.4000 |
| recall_any@10 | 1.0000 |
| recall_any@3 | 1.0000 |
| recall_any@5 | 1.0000 |

## Timing Summary

| Metric | Value (s) |
|--------|----------:|
| mean_cleanup_seconds | 0.0000 |
| mean_ingest_seconds | 0.0001 |
| mean_recall_seconds | 0.0001 |
| total_cleanup_seconds | 0.0001 |
| total_ingest_seconds | 0.0003 |
| total_recall_seconds | 0.0004 |

## Configuration

```json
{
  "benchmark": "chinese",
  "split": "test",
  "limit": 5,
  "output_dir": "full_test_results/bm25_chinese",
  "adapter": {
    "name": "bm25",
    "options": {}
  },
  "judge": {
    "enabled": true,
    "model": "openai/gpt-4o-mini",
    "provider": "openai",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://openrouter.ai/api/v1"
  },
  "metrics": {
    "retrieval_k": [
      1,
      3,
      5,
      10
    ],
    "include_latency": true,
    "include_cost": false,
    "compute_semantic": false,
    "semantic_retrieval_k": [
      1,
      3,
      5,
      10
    ],
    "semantic_judge_model": "claude-haiku-4-5-20251001"
  },
  "reporting": {
    "formats": [
      "console",
      "json",
      "markdown",
      "html"
    ]
  }
}
```
