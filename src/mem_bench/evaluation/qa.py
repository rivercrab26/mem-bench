"""QA evaluation using LLM-as-Judge."""

from __future__ import annotations

from mem_bench.core.judge import Judge
from mem_bench.core.types import RecallResult


def generate_answer(
    question: str,
    context: str,
    question_date: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Generate an answer using LLM with retrieved context."""
    try:
        import os
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install mem-bench[judge]")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=key, base_url=base_url)

    system_msg = (
        "You are a helpful assistant with access to your memory of past conversations. "
        "Answer the user's question based on the relevant memories provided. "
        "If the information is not available in the memories, say you don't have that information."
    )

    user_msg = (
        "Based on the relevant memories below, answer the question concisely.\n\n"
        f"Relevant Memories:\n{context}\n\n"
        f"Current Date: {question_date}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=500,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def format_recall_context(results: list[RecallResult]) -> str:
    """Format recalled memories into a context string for answer generation."""
    if not results:
        return "(No relevant memories found.)"

    parts = []
    for i, r in enumerate(results, 1):
        header = f"Memory {i}"
        if r.document_id:
            header += f" (source: {r.document_id})"
        parts.append(f"[{header}]\n{r.content}")

    return "\n\n".join(parts)


def evaluate_qa(
    judge: Judge,
    question: str,
    reference_answer: str,
    hypothesis: str,
    question_type: str,
    sample_id: str = "",
) -> bool:
    """Evaluate a single QA pair using the judge."""
    is_abstention = sample_id.endswith("_abs")
    return judge.evaluate(
        question=question,
        reference=reference_answer,
        hypothesis=hypothesis,
        question_type=question_type,
        is_abstention=is_abstention,
    )
