"""Retrieval evaluation metrics.

Computes recall, NDCG, and MRR from a list of ``RecallResult`` objects
against ground-truth document IDs.

Also provides semantic retrieval metrics that use LLM or token-overlap
to evaluate recall quality for systems that extract facts rather than
preserving original document IDs.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Sequence

from mem_bench.core.types import RecallResult

logger = logging.getLogger(__name__)


def _unique_doc_ids(results: Sequence[RecallResult]) -> list[str]:
    """Extract an ordered, deduplicated list of document IDs from results."""
    seen: set[str] = set()
    ids: list[str] = []
    for r in results:
        if r.document_id and r.document_id not in seen:
            seen.add(r.document_id)
            ids.append(r.document_id)
    return ids


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at *k*.

    Uses the standard formula: rel_i / log2(i + 2) for 0-indexed positions,
    which is equivalent to rel_i / log2(rank + 1) where rank starts at 1.
    """
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += rel / math.log2(i + 2)
    return total


def _ndcg_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at *k*.

    Each retrieved document gets relevance 1 if it is in the ground truth,
    else 0.  The ideal ranking puts all relevant documents first.
    """
    relevances = [1.0 if doc_id in ground_truth_ids else 0.0 for doc_id in retrieved_ids[:k]]
    actual = _dcg(relevances, k)

    # Ideal: all relevant docs at the top
    n_relevant = min(len(ground_truth_ids), k)
    ideal_relevances = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    ideal = _dcg(ideal_relevances, k)

    if ideal == 0.0:
        return 0.0
    return actual / ideal


def _mrr(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Mean Reciprocal Rank (for a single query).

    Returns 1/rank of the first relevant document, or 0 if none is found.
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / i
    return 0.0


def compute_retrieval_metrics(
    results: list[RecallResult],
    ground_truth_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics for a single query.

    Args:
        results: Recall results returned by the adapter, ordered by relevance.
        ground_truth_ids: List of document IDs that should be retrieved.
        k_values: Cut-off values for recall/NDCG (default ``[1, 3, 5, 10]``).

    Returns:
        Dictionary of metric names to values, e.g.::

            {
                "recall_any@1": 1.0,
                "recall_all@1": 0.0,
                "ndcg@1": 1.0,
                "recall_any@3": 1.0,
                "recall_all@3": 1.0,
                "ndcg@3": 0.87,
                ...
                "mrr": 0.5,
            }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    gt_set = set(ground_truth_ids)
    retrieved = _unique_doc_ids(results)

    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = set(retrieved[:k])

        # recall_any@k: at least one relevant doc in top-k
        recall_any = 1.0 if top_k & gt_set else 0.0
        # recall_all@k: all relevant docs in top-k
        recall_all = 1.0 if gt_set.issubset(top_k) else 0.0

        ndcg_score = _ndcg_at_k(retrieved, gt_set, k)

        metrics[f"recall_any@{k}"] = recall_any
        metrics[f"recall_all@{k}"] = recall_all
        metrics[f"ndcg@{k}"] = ndcg_score

    metrics["mrr"] = _mrr(retrieved, gt_set)

    return metrics


# ---------------------------------------------------------------------------
# Semantic evaluation helpers
# ---------------------------------------------------------------------------


def semantic_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard overlap between two texts.

    This is a fast, non-LLM fallback for estimating whether two pieces of
    text cover similar information.

    Returns:
        Float in [0, 1] representing the Jaccard similarity of the token sets.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _get_llm_client():
    """Create an LLM client using Anthropic API with fallback to OpenAI-compatible.

    Uses environment variables:
    - ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY for Anthropic
    - ANTHROPIC_BASE_URL for custom Anthropic endpoint
    - Falls back to OpenAI-compatible via OPENAI_API_KEY / OPENAI_BASE_URL

    Returns:
        Tuple of (client, provider) where provider is 'anthropic' or 'openai'.
    """
    anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    )
    if anthropic_key:
        try:
            import anthropic

            kwargs = {"api_key": anthropic_key}
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs), "anthropic"
        except ImportError:
            logger.debug("anthropic package not installed, trying openai fallback")

    # Fallback to OpenAI-compatible
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI

            kwargs = {"api_key": openai_key}
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs), "openai"
        except ImportError:
            pass

    raise RuntimeError(
        "Semantic evaluation requires either the anthropic or openai package "
        "and a corresponding API key (ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY)."
    )


def _llm_yes_no(prompt: str, model: str) -> bool:
    """Ask an LLM a yes/no question. Returns True if the answer contains 'yes'."""
    client, provider = _get_llm_client()

    if provider == "anthropic":
        msg = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().lower()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().lower()

    return "yes" in text


def compute_semantic_retrieval_metrics(
    recall_results: list[RecallResult],
    question: str,
    reference_answer: str,
    ground_truth_contents: list[str],
    k_values: list[int] | None = None,
    judge_model: str = "claude-haiku-4-5-20251001",
) -> dict[str, float]:
    """Compute semantic retrieval metrics using LLM judgement.

    Unlike :func:`compute_retrieval_metrics` which relies on exact
    ``document_id`` matching, this function uses an LLM to judge whether
    the recalled content semantically contains the information needed to
    answer the question.  This is fairer for fact-extraction systems
    (e.g. Mem0, Graphiti) that decompose documents into extracted facts
    and may not preserve the original document ID.

    Args:
        recall_results: Recall results returned by the adapter, ordered by
            relevance (best first).
        question: The benchmark question being evaluated.
        reference_answer: The gold reference answer.
        ground_truth_contents: Actual text of the ground-truth sessions /
            documents that should be retrieved.
        k_values: Cut-off values for the metrics (default ``[1, 3, 5, 10]``).
        judge_model: LLM model identifier used for semantic judgement.

    Returns:
        Dictionary with metric names to float values::

            {
                "semantic_recall@1": 0 or 1,
                "semantic_recall@3": 0 or 1,
                ...
                "content_coverage@1": 0.5,
                "content_coverage@3": 1.0,
                ...
            }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: dict[str, float] = {}

    for k in k_values:
        top_k_results = recall_results[:k]
        if not top_k_results:
            metrics[f"semantic_recall@{k}"] = 0.0
            metrics[f"content_coverage@{k}"] = 0.0
            continue

        # Combine recalled content for the semantic recall check
        recalled_text = "\n\n---\n\n".join(r.content for r in top_k_results if r.content)

        # --- semantic_recall@k: can we answer the question? ---
        recall_prompt = (
            "Given the following recalled memories, could you answer the "
            f"question '{question}' with the correct answer '{reference_answer}'? "
            "Answer yes or no.\n\n"
            f"Recalled memories:\n{recalled_text}"
        )
        try:
            semantic_hit = _llm_yes_no(recall_prompt, judge_model)
        except Exception:
            logger.warning("LLM call failed for semantic_recall@%d, defaulting to 0", k)
            semantic_hit = False
        metrics[f"semantic_recall@{k}"] = 1.0 if semantic_hit else 0.0

        # --- content_coverage@k: fraction of ground truths covered ---
        if not ground_truth_contents:
            metrics[f"content_coverage@{k}"] = 0.0
            continue

        covered = 0
        for gt_content in ground_truth_contents:
            coverage_prompt = (
                "Does the following recalled text contain the same key information "
                "as the reference text below? Answer yes or no.\n\n"
                f"Recalled text:\n{recalled_text}\n\n"
                f"Reference text:\n{gt_content}"
            )
            try:
                is_covered = _llm_yes_no(coverage_prompt, judge_model)
            except Exception:
                logger.warning("LLM call failed for content_coverage, defaulting to not covered")
                is_covered = False
            if is_covered:
                covered += 1
        metrics[f"content_coverage@{k}"] = covered / len(ground_truth_contents)

    return metrics
