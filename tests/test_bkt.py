from __future__ import annotations

import pytest
from pydantic import ValidationError

from swetrack.ml.knowledge_tracing.bkt import (
    DEFAULT_PARAMETERS,
    BKTParameters,
    predicted_correct_probability,
    trace_mastery,
    update_mastery,
)


def test_default_parameters_are_valid_probabilities():
    for value in (
        DEFAULT_PARAMETERS.p_init,
        DEFAULT_PARAMETERS.p_transit,
        DEFAULT_PARAMETERS.p_guess,
        DEFAULT_PARAMETERS.p_slip,
    ):
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("field", ["p_init", "p_transit", "p_guess", "p_slip"])
def test_parameters_reject_out_of_range_values(field):
    kwargs = {"p_init": 0.3, "p_transit": 0.1, "p_guess": 0.2, "p_slip": 0.1}
    kwargs[field] = 1.5
    with pytest.raises(ValidationError):
        BKTParameters(**kwargs)


def test_correct_answer_increases_mastery_under_default_parameters():
    prior = 0.3
    posterior = update_mastery(prior, correct=True, params=DEFAULT_PARAMETERS)
    assert posterior > prior


def test_incorrect_answer_decreases_mastery_under_default_parameters():
    prior = 0.3
    posterior = update_mastery(prior, correct=False, params=DEFAULT_PARAMETERS)
    assert posterior < prior


def test_mastery_stays_within_unit_interval_over_a_long_sequence():
    outcomes = [True, False, True, True, False, False, True, False, True, True] * 5
    history = trace_mastery(outcomes, DEFAULT_PARAMETERS)
    assert all(0.0 <= value <= 1.0 for value in history)
    assert len(history) == len(outcomes)


def test_trace_mastery_is_deterministic():
    outcomes = [True, False, True, True, False]
    first = trace_mastery(outcomes, DEFAULT_PARAMETERS)
    second = trace_mastery(outcomes, DEFAULT_PARAMETERS)
    assert first == second


def test_trace_mastery_matches_sequential_update_mastery_calls():
    outcomes = [True, False, True]
    mastery = DEFAULT_PARAMETERS.p_init
    manual_history = []
    for correct in outcomes:
        mastery = update_mastery(mastery, correct, DEFAULT_PARAMETERS)
        manual_history.append(mastery)
    assert trace_mastery(outcomes, DEFAULT_PARAMETERS) == manual_history


def test_update_mastery_is_numerically_safe_at_degenerate_parameters():
    degenerate = BKTParameters(p_init=0.0, p_transit=0.0, p_guess=0.0, p_slip=0.0)
    # prior=0, correct=True, p_guess=0 -> denominator is exactly zero.
    result = update_mastery(0.0, correct=True, params=degenerate)
    assert result == 0.0  # falls back to the prior rather than raising


def test_predicted_correct_probability_bounds():
    for mastery in (0.0, 0.3, 0.7, 1.0):
        p = predicted_correct_probability(mastery, DEFAULT_PARAMETERS)
        assert 0.0 <= p <= 1.0

    # Fully mastered: P(correct) should equal 1 - p_slip.
    assert predicted_correct_probability(1.0, DEFAULT_PARAMETERS) == pytest.approx(1.0 - DEFAULT_PARAMETERS.p_slip)
    # No mastery at all: P(correct) should equal p_guess.
    assert predicted_correct_probability(0.0, DEFAULT_PARAMETERS) == pytest.approx(DEFAULT_PARAMETERS.p_guess)
