# Benchmark Report: Supermemory

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | oracle |
| Samples | 5 |
| Failed | 0 |
| Total Time | 218.1s |

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal-reasoning | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8000 |
| **Overall** | 5 | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.8000** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 0.0000 |
| ndcg@1 | 0.0000 |
| ndcg@10 | 0.0000 |
| ndcg@3 | 0.0000 |
| ndcg@5 | 0.0000 |
| qa_accuracy | 0.8000 |
| recall_all@1 | 0.0000 |
| recall_all@10 | 0.0000 |
| recall_all@3 | 0.0000 |
| recall_all@5 | 0.0000 |
| recall_any@1 | 0.0000 |
| recall_any@10 | 0.0000 |
| recall_any@3 | 0.0000 |
| recall_any@5 | 0.0000 |

## Timing Summary

| Metric | Value (s) |
|--------|----------:|
| mean_cleanup_seconds | 2.6231 |
| mean_ingest_seconds | 35.8341 |
| mean_recall_seconds | 1.8290 |
| total_cleanup_seconds | 13.1157 |
| total_ingest_seconds | 179.1704 |
| total_recall_seconds | 9.1450 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "oracle",
  "limit": 5,
  "output_dir": "full_test_results/supermemory_oracle_5_v2",
  "adapter": {
    "name": "supermemory",
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
