# Benchmark Report: BM25

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | s |
| Samples | 20 |
| Failed | 0 |
| Total Time | 151.1s |

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8500 |
| **Overall** | 20 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.8500** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 1.0000 |
| ndcg@1 | 1.0000 |
| ndcg@10 | 1.0000 |
| ndcg@3 | 1.0000 |
| ndcg@5 | 1.0000 |
| qa_accuracy | 0.8500 |
| recall_all@1 | 1.0000 |
| recall_all@10 | 1.0000 |
| recall_all@3 | 1.0000 |
| recall_all@5 | 1.0000 |
| recall_any@1 | 1.0000 |
| recall_any@10 | 1.0000 |
| recall_any@3 | 1.0000 |
| recall_any@5 | 1.0000 |

## Timing Summary

| Metric | Value (s) |
|--------|----------:|
| mean_cleanup_seconds | 0.0019 |
| mean_ingest_seconds | 0.0259 |
| mean_recall_seconds | 0.0002 |
| total_cleanup_seconds | 0.0389 |
| total_ingest_seconds | 0.5182 |
| total_recall_seconds | 0.0038 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "s",
  "limit": 20,
  "output_dir": "full_test_results/bm25_s_20",
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
