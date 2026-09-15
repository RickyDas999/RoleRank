from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from swetrack.domains.learning.services import create_activity, get_study_recommendations, record_attempt


def _create_activity(session, **overrides):
    defaults = dict(
        slug="graph-practice", title="Graph Practice", activity_type="concept_review", skill_ids=["graphs"]
    )
    defaults.update(overrides)
    return create_activity(session, **defaults)


def test_never_practiced_activity_outranks_a_freshly_mastered_one(db_session):
    _create_activity(db_session, slug="graph-practice", title="Graph Practice", skill_ids=["graphs"])
    strong_activity = _create_activity(db_session, slug="tree-practice", title="Tree Practice", skill_ids=["trees"])

    for _ in range(5):
        record_attempt(db_session, activity_id=strong_activity.id, success=True)

    now = datetime.now(timezone.utc)
    recommendations = get_study_recommendations(db_session, now=now)

    ranked_slugs = [rec.activity.slug for rec in recommendations]
    assert ranked_slugs.index("graph-practice") < ranked_slugs.index("tree-practice")


def test_never_practiced_skill_has_maximal_gap_and_staleness(db_session):
    activity = _create_activity(db_session)

    recommendations = get_study_recommendations(db_session)

    rec = next(r for r in recommendations if r.activity.id == activity.id)
    assert rec.components.mastery_gap == pytest.approx(0.7)  # 1 - default BKT p_init (0.3)
    assert rec.components.staleness == 1.0


def test_staleness_increases_with_simulated_elapsed_time(db_session):
    activity = _create_activity(db_session)
    record_attempt(db_session, activity_id=activity.id, success=True)
    recorded_at = datetime.now(timezone.utc)

    soon_after = get_study_recommendations(db_session, now=recorded_at)
    much_later = get_study_recommendations(db_session, now=recorded_at + timedelta(days=60))

    soon_rec = next(r for r in soon_after if r.activity.id == activity.id)
    later_rec = next(r for r in much_later if r.activity.id == activity.id)
    assert soon_rec.components.staleness < later_rec.components.staleness


def test_repetition_penalty_reflects_recent_attempts_on_same_activity(db_session):
    activity = _create_activity(db_session)
    now = datetime.now(timezone.utc)
    for _ in range(3):
        record_attempt(db_session, activity_id=activity.id, success=True)

    recommendations = get_study_recommendations(db_session, now=now)

    rec = next(r for r in recommendations if r.activity.id == activity.id)
    assert rec.components.repetition_penalty > 0.0


def test_repetition_penalty_ignores_attempts_older_than_seven_days(db_session):
    activity = _create_activity(db_session)
    record_attempt(db_session, activity_id=activity.id, success=True)

    far_future = datetime.now(timezone.utc) + timedelta(days=30)
    recommendations = get_study_recommendations(db_session, now=far_future)

    rec = next(r for r in recommendations if r.activity.id == activity.id)
    assert rec.components.repetition_penalty == 0.0


def test_top_k_limits_result_count(db_session):
    for i in range(3):
        _create_activity(db_session, slug=f"activity-{i}", title=f"Activity {i}", skill_ids=["graphs"])

    recommendations = get_study_recommendations(db_session, top_k=2)

    assert len(recommendations) == 2


def test_results_are_sorted_descending_by_score(db_session):
    _create_activity(db_session, slug="a", title="A", skill_ids=["graphs"])
    _create_activity(db_session, slug="b", title="B", skill_ids=["trees"])

    recommendations = get_study_recommendations(db_session)

    scores = [rec.score for rec in recommendations]
    assert scores == sorted(scores, reverse=True)
