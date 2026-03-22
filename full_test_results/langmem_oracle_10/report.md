# Benchmark Report: LangMem

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | oracle |
| Samples | 10 |
| Failed | 0 |
| Total Time | 99.1s |

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal-reasoning | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8000 |
| **Overall** | 10 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.8000** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 1.0000 |
| ndcg@1 | 1.0000 |
| ndcg@10 | 1.0000 |
| ndcg@3 | 1.0000 |
| ndcg@5 | 1.0000 |
| qa_accuracy | 0.8000 |
| recall_all@1 | 0.0000 |
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
| mean_cleanup_seconds | 0.0000 |
| mean_ingest_seconds | 3.2711 |
| mean_recall_seconds | 1.0571 |
| total_cleanup_seconds | 0.0005 |
| total_ingest_seconds | 32.7107 |
| total_recall_seconds | 10.5705 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "oracle",
  "limit": 10,
  "output_dir": "full_test_results/langmem_oracle_10",
  "adapter": {
    "name": "langmem",
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
    "include_cost": false
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
