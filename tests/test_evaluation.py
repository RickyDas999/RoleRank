"""Tests for Precision@K and NDCG@K against hand-calculated examples."""

from __future__ import annotations

import pytest

from rolerank.evaluation import ndcg_at_k, precision_at_k

# Hand-calculated example:
# labels: A=2, B=1, C=0, D=2 (graded relevance)
# ranking: A, C, B, D
_LABELS = {"A": 2, "B": 1, "C": 0, "D": 2}
_RANKED = ["A", "C", "B", "D"]


def test_precision_at_k_hand_calculated():
    # top-3 = [A, C, B]; relevant (label > 0): A, B -> 2/3
    assert precision_at_k(_RANKED, _LABELS, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_all_relevant():
    labels = {"A": 1, "B": 2}
    assert precision_at_k(["A", "B"], labels, k=2) == pytest.approx(1.0)


def test_precision_at_k_none_relevant():
    labels = {"A": 0, "B": 0}
    assert precision_at_k(["A", "B"], labels, k=2) == pytest.approx(0.0)


def test_precision_at_k_missing_job_id_treated_as_irrelevant():
    labels = {"A": 2}
    assert precision_at_k(["A", "UNKNOWN"], labels, k=2) == pytest.approx(0.5)


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(_RANKED, _LABELS, k=0)


def test_ndcg_at_k_hand_calculated():
    # DCG@3 = (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4) = 3 + 0 + 0.5 = 3.5
    # Ideal ranking by label desc: [2, 2, 1] ->
    # IDCG@3 = 3/log2(2) + 3/log2(3) + 1/log2(4) = 3 + 1.892789... + 0.5 = 5.392789...
    # NDCG@3 = 3.5 / 5.392789... = 0.649014...
    assert ndcg_at_k(_RANKED, _LABELS, k=3) == pytest.approx(0.649014791936513, rel=1e-9)


def test_ndcg_at_k_perfect_ranking_is_one():
    labels = {"A": 2, "B": 1, "C": 0}
    assert ndcg_at_k(["A", "B", "C"], labels, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_zero_ideal_dcg_returns_zero():
    labels = {"A": 0, "B": 0}
    assert ndcg_at_k(["A", "B"], labels, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        ndcg_at_k(_RANKED, _LABELS, k=0)
