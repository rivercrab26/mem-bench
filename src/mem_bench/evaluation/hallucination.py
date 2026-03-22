"""Hallucination metrics for HaluMem benchmark."""

from __future__ import annotations


def compute_hallucination_metrics(
    extracted_memories: list[str],
    gold_memories: list[str],
    updated_memories: list[str] | None = None,
    gold_updates: list[str] | None = None,
) -> dict[str, float]:
    """Compute HaluMem-style hallucination metrics.

    Returns:
        fabrication_rate: Memories with no basis in source
        error_rate: Memories that semantically deviate from gold
        omission_rate: Gold memories that were missed
    """
    # Placeholder - full implementation requires LLM-based semantic matching
    # For now, compute simple overlap metrics
    if not gold_memories:
        return {"fabrication_rate": 0.0, "error_rate": 0.0, "omission_rate": 0.0}

    gold_set = set(gold_memories)
    extracted_set = set(extracted_memories)

    matched = gold_set & extracted_set
    fabricated = extracted_set - gold_set
    omitted = gold_set - extracted_set

    total_extracted = len(extracted_set) or 1
    total_gold = len(gold_set) or 1

    return {
        "fabrication_rate": len(fabricated) / total_extracted,
        "error_rate": 0.0,  # Requires semantic comparison
        "omission_rate": len(omitted) / total_gold,
    }
