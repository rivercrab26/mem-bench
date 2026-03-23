# Benchmark Report: Supermemory

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | oracle |
| Samples | 20 |
| Failed | 0 |
| Total Time | 349.5s |

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal-reasoning | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0500 |
| **Overall** | 20 | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0500** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 0.0000 |
| ndcg@1 | 0.0000 |
| ndcg@10 | 0.0000 |
| ndcg@3 | 0.0000 |
| ndcg@5 | 0.0000 |
| qa_accuracy | 0.0500 |
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
| mean_cleanup_seconds | 2.7037 |
| mean_ingest_seconds | 10.3080 |
| mean_recall_seconds | 1.7975 |
| total_cleanup_seconds | 54.0746 |
| total_ingest_seconds | 206.1607 |
| total_recall_seconds | 35.9507 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "oracle",
  "limit": 20,
  "output_dir": "full_test_results/supermemory_oracle_20",
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
