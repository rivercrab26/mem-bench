# Benchmark Report: Letta

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | oracle |
| Samples | 10 |
| Failed | 10 |
| Total Time | 0.1s |

## Timing Summary

| Metric | Value (s) |
|--------|----------:|
| mean_cleanup_seconds | 0.0000 |
| mean_ingest_seconds | 0.0000 |
| mean_recall_seconds | 0.0000 |
| total_cleanup_seconds | 0.0000 |
| total_ingest_seconds | 0.0000 |
| total_recall_seconds | 0.0000 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "oracle",
  "limit": 10,
  "output_dir": "full_test_results/letta_oracle_10",
  "adapter": {
    "name": "letta",
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
