from __future__ import annotations

import pytest

from swetrack.ml.evaluation.knowledge_tracing import walk_forward_evaluate
from swetrack.ml.knowledge_tracing.bkt import DEFAULT_PARAMETERS


def test_walk_forward_evaluate_requires_at_least_two_outcomes():
    with pytest.raises(ValueError, match="at least 2 outcomes"):
        walk_forward_evaluate([True], DEFAULT_PARAMETERS)


def test_walk_forward_evaluate_evaluates_one_fewer_than_total_outcomes():
    outcomes = [True, False, True, True, False, True]
    result = walk_forward_evaluate(outcomes, DEFAULT_PARAMETERS)
    assert result["n_evaluated"] == len(outcomes) - 1


def test_walk_forward_evaluate_returns_nonnegative_scores():
    outcomes = [True, False, True, True, False, False, True, True, True, False]
    result = walk_forward_evaluate(outcomes, DEFAULT_PARAMETERS)
    for model_key in ("bkt", "baseline_historical_success_rate"):
        assert result[model_key]["brier_score"] >= 0.0
        assert result[model_key]["log_loss"] >= 0.0


def test_walk_forward_evaluate_is_deterministic():
    outcomes = [False, False, False, True, True, True, True, True]
    first = walk_forward_evaluate(outcomes, DEFAULT_PARAMETERS)
    second = walk_forward_evaluate(outcomes, DEFAULT_PARAMETERS)
    assert first == second
