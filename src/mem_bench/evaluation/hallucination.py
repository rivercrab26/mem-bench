"""Hallucination metrics for HaluMem benchmark.

Uses LLM to semantically compare extracted memories against gold references.
Falls back to token-overlap matching when LLM is unavailable.
"""

from __future__ import annotations

from mem_bench.core.types import RecallResult
from mem_bench.evaluation.retrieval import _get_llm_client


def compute_hallucination_metrics(
    recall_results: list[RecallResult],
    ground_truth_contents: list[str],
    judge_model: str = "claude-haiku-4-5-20251001",
) -> dict[str, float]:
    """Compute HaluMem-style hallucination metrics.

    Metrics:
    - integrity (recall): fraction of gold memories found in retrieved results
    - accuracy (precision): fraction of retrieved results matching a gold memory
    - fabrication_rate: 1 - accuracy
    - omission_rate: 1 - integrity
    """
    if not ground_truth_contents:
        return {
            "integrity": 1.0,
            "accuracy": 1.0,
            "fabrication_rate": 0.0,
            "omission_rate": 0.0,
        }
    if not recall_results:
        return {
            "integrity": 0.0,
            "accuracy": 0.0,
            "fabrication_rate": 0.0,
            "omission_rate": 1.0,
        }

    try:
        return _llm_hallucination_metrics(
            recall_results, ground_truth_contents, judge_model
        )
    except Exception:
        return _fuzzy_hallucination_metrics(recall_results, ground_truth_contents)


def _llm_yes_no(client, provider: str, prompt: str, model: str) -> bool:
    """Ask an LLM a yes/no question. Returns True if the answer contains 'yes'."""
    if provider == "anthropic":
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().lower()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        text = response.choices[0].message.content.strip().lower()

    return "yes" in text


def _llm_hallucination_metrics(
    results: list[RecallResult],
    gold: list[str],
    model: str,
) -> dict[str, float]:
    """Use LLM to check semantic equivalence.

    Uses _get_llm_client() from retrieval.py which supports both Anthropic
    and OpenAI-compatible providers.
    """
    client, provider = _get_llm_client()

    retrieved_texts = [r.content for r in results[:10]]
    retrieved_block = "\n---\n".join(retrieved_texts)
    gold_block = "\n---\n".join(gold)

    # Check integrity: how many gold memories are covered
    gold_found = 0
    for g in gold:
        prompt = (
            "Does any of the following retrieved memories contain "
            "the same information as the reference?\n\n"
            f"Reference: {g}\n\n"
            f"Retrieved memories:\n{retrieved_block}\n\n"
            "Answer yes or no only."
        )
        if _llm_yes_no(client, provider, prompt, model):
            gold_found += 1

    # Check accuracy: how many retrieved results match some gold memory
    accurate = 0
    for r_text in retrieved_texts:
        prompt = (
            "Does this retrieved memory match any of the "
            "reference facts?\n\n"
            f"Retrieved: {r_text}\n\n"
            f"Reference facts:\n{gold_block}\n\n"
            "Answer yes or no only."
        )
        if _llm_yes_no(client, provider, prompt, model):
            accurate += 1

    n_gold = len(gold)
    n_retrieved = len(retrieved_texts)
    integrity = gold_found / n_gold if n_gold else 1.0
    accuracy = accurate / n_retrieved if n_retrieved else 1.0

    return {
        "integrity": integrity,
        "accuracy": accuracy,
        "fabrication_rate": 1.0 - accuracy,
        "omission_rate": 1.0 - integrity,
    }


def _fuzzy_hallucination_metrics(
    results: list[RecallResult],
    gold: list[str],
) -> dict[str, float]:
    """Fallback: use token overlap for approximate matching."""
    retrieved_tokens = [set(r.content.lower().split()) for r in results[:10]]
    gold_token_sets = [set(g.lower().split()) for g in gold]

    gold_found = 0
    for g_set in gold_token_sets:
        for r_set in retrieved_tokens:
            overlap = len(g_set & r_set) / max(len(g_set), 1)
            if overlap > 0.3:
                gold_found += 1
                break

    accurate = 0
    for r_set in retrieved_tokens:
        for g_set in gold_token_sets:
            overlap = len(g_set & r_set) / max(len(r_set), 1)
            if overlap > 0.3:
                accurate += 1
                break

    n_gold = len(gold)
    n_ret = len(retrieved_tokens)
    integrity = gold_found / n_gold if n_gold else 1.0
    accuracy = accurate / n_ret if n_ret else 1.0

    return {
        "integrity": integrity,
        "accuracy": accuracy,
        "fabrication_rate": 1.0 - accuracy,
        "omission_rate": 1.0 - integrity,
    }
