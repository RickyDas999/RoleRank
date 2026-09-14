"""Offline ranking evaluation: Precision@K and NDCG@K from relevance labels.

Relevance labels are ordinal (0/1/2), manually curated for one example
candidate profile. They are demonstration judgments, not objective ground
truth (see docs/ml-design.md).
"""

from __future__ import annotations

import math


def precision_at_k(ranked_job_ids: list[str], relevance_labels: dict[str, int], k: int) -> float:
    """Fraction of the top-K ranked jobs with a relevance label greater than zero."""
    if k < 1:
        raise ValueError("k must be >= 1")
    top_k = ranked_job_ids[:k]
    relevant = sum(1 for job_id in top_k if relevance_labels.get(job_id, 0) > 0)
    return relevant / k


def _dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain for a relevance sequence in rank order."""
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_job_ids: list[str], relevance_labels: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain at K using graded relevance labels.

    Returns 0.0 when the ideal DCG is zero (no relevant labels exist),
    rather than dividing by zero.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    top_k = ranked_job_ids[:k]
    relevances = [relevance_labels.get(job_id, 0) for job_id in top_k]
    dcg = _dcg(relevances)

    ideal_relevances = sorted(relevance_labels.values(), reverse=True)[:k]
    idcg = _dcg(ideal_relevances)

    if idcg == 0:
        return 0.0
    return dcg / idcg
