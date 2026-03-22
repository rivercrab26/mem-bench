"""LLM-as-Judge interface for QA evaluation."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Interface for LLM-based answer evaluation."""

    def evaluate(self, question: str, reference: str, hypothesis: str,
                 question_type: str, is_abstention: bool = False) -> bool:
        """Return True if the hypothesis is correct."""
        ...


class OpenAIJudge:
    """Judge using OpenAI-compatible API (supports OpenAI, Anthropic via proxy, local)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for LLM-as-Judge. "
                "Install with: pip install mem-bench[judge]"
            )

        self.model = model
        key = api_key or os.environ.get(api_key_env, "")
        self._client = OpenAI(api_key=key, base_url=base_url)

    def evaluate(
        self,
        question: str,
        reference: str,
        hypothesis: str,
        question_type: str,
        is_abstention: bool = False,
    ) -> bool:
        prompt = _build_judge_prompt(question_type, question, reference, hypothesis, is_abstention)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().lower()
        return "yes" in text


def _build_judge_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Build evaluation prompt per question type.

    Ported from LongMemEval's evaluate_qa.py to maintain result comparability.
    """
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the question "
            "as unanswerable. The model could say that the information is incomplete, or some "
            "other information is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. \n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. "
            "In addition, do not penalize off-by-one errors for the number of days. If the "
            "question asks for the number of days/weeks/months, etc., and the model makes "
            "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
            "response is still correct. \n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is "
            "the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a "
            "response from a model. Please answer yes if the response satisfies the desired "
            "response. Otherwise, answer no. The model does not need to reflect all the points "
            "in the rubric. The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    # Fallback for unknown types
    return (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )
