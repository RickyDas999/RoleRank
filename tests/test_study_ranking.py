from __future__ import annotations

import pytest

from swetrack.ml.study_ranking.ranker import (
    DEFAULT_WEIGHTS,
    StudyRankingComponents,
    compute_difficulty_fit,
    compute_mastery_gap,
    compute_repetition_penalty,
    compute_staleness,
    score_activity,
)


def test_compute_mastery_gap_is_inverse_of_mastery():
    assert compute_mastery_gap(0.0) == 1.0
    assert compute_mastery_gap(1.0) == 0.0
    assert compute_mastery_gap(0.7) == pytest.approx(0.3)


def test_compute_staleness_never_practiced_is_maximally_stale():
    assert compute_staleness(None) == 1.0


def test_compute_staleness_increases_monotonically_with_days():
    fresh = compute_staleness(0.0)
    a_week = compute_staleness(7.0)
    a_month = compute_staleness(30.0)
    assert fresh == 0.0
    assert fresh < a_week < a_month
    assert a_month < 1.0


def test_compute_difficulty_fit_prefers_easy_for_high_mastery_gap():
    high_gap = 0.9
    assert compute_difficulty_fit(high_gap, "easy") > compute_difficulty_fit(high_gap, "hard")


def test_compute_difficulty_fit_prefers_hard_for_low_mastery_gap():
    low_gap = 0.1
    assert compute_difficulty_fit(low_gap, "hard") > compute_difficulty_fit(low_gap, "easy")


def test_compute_difficulty_fit_is_neutral_when_undeclared():
    assert compute_difficulty_fit(0.9, None) == 0.5


def test_compute_repetition_penalty_saturates_at_cap():
    assert compute_repetition_penalty(0) == 0.0
    assert compute_repetition_penalty(5, cap=5) == 1.0
    assert compute_repetition_penalty(100, cap=5) == 1.0


def test_score_activity_rewards_high_gap_and_staleness():
    weak_and_stale = StudyRankingComponents(mastery_gap=0.9, staleness=0.9, difficulty_fit=0.8, repetition_penalty=0.0)
    strong_and_fresh = StudyRankingComponents(
        mastery_gap=0.1, staleness=0.1, difficulty_fit=0.8, repetition_penalty=0.0
    )
    assert score_activity(weak_and_stale) > score_activity(strong_and_fresh)


def test_score_activity_penalizes_recent_repetition():
    base = StudyRankingComponents(mastery_gap=0.8, staleness=0.8, difficulty_fit=0.8, repetition_penalty=0.0)
    drilled = StudyRankingComponents(mastery_gap=0.8, staleness=0.8, difficulty_fit=0.8, repetition_penalty=1.0)
    assert score_activity(drilled, DEFAULT_WEIGHTS) < score_activity(base, DEFAULT_WEIGHTS)


def test_score_activity_stays_in_unit_range():
    extreme = StudyRankingComponents(mastery_gap=1.0, staleness=1.0, difficulty_fit=1.0, repetition_penalty=0.0)
    assert 0.0 <= score_activity(extreme) <= 1.0
