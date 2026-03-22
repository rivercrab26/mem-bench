"""Token usage and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass

# Approximate costs per 1M tokens (USD) as of 2026
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


@dataclass
class TokenUsage:
    """Track token usage across a benchmark run."""

    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    gen_input_tokens: int = 0
    gen_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.judge_input_tokens
            + self.judge_output_tokens
            + self.gen_input_tokens
            + self.gen_output_tokens
        )

    def estimated_cost(self, model: str) -> float:
        costs = MODEL_COSTS.get(model, {"input": 1.0, "output": 3.0})
        input_tokens = self.judge_input_tokens + self.gen_input_tokens
        output_tokens = self.judge_output_tokens + self.gen_output_tokens
        input_cost = input_tokens / 1_000_000 * costs["input"]
        output_cost = output_tokens / 1_000_000 * costs["output"]
        return input_cost + output_cost

    def to_dict(self) -> dict[str, float]:
        return {
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "gen_input_tokens": self.gen_input_tokens,
            "gen_output_tokens": self.gen_output_tokens,
            "total_tokens": self.total_tokens,
        }
