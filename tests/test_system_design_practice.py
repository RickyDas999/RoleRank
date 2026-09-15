from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from swetrack.domains.learning.models import AttemptRecord, SkillEventRecord
from swetrack.domains.learning.services import (
    create_activity,
    get_mastery,
    get_skill_events,
    get_system_design_attempt,
    record_system_design_attempt,
)

_TICKETMASTER_SKILLS = ["requirements", "capacity-estimation", "consistency", "reliability", "tradeoff-reasoning"]


def _create_system_design_activity(session, **overrides):
    defaults = dict(
        slug="design-ticketmaster",
        title="Design Ticketmaster",
        activity_type="system_design",
        skill_ids=list(_TICKETMASTER_SKILLS),
    )
    defaults.update(overrides)
    return create_activity(session, **defaults)


def test_record_system_design_attempt_stores_per_skill_scores(db_session):
    activity = _create_system_design_activity(db_session)
    scores = {
        "requirements": 0.9,
        "capacity-estimation": 0.6,
        "consistency": 0.5,
        "reliability": 0.4,
        "tradeoff-reasoning": 0.7,
    }

    attempt, result, events = record_system_design_attempt(db_session, activity_id=activity.id, scores=scores)

    assert result.attempt_id == attempt.id
    assert result.scores == scores
    assert {event.skill_id: event.outcome for event in events} == scores
    assert all(event.source_type == "system_design_attempt" for event in events)


def test_record_system_design_attempt_updates_each_skill_mastery_independently(db_session):
    activity = _create_system_design_activity(db_session)
    scores = {
        "requirements": 0.9,
        "capacity-estimation": 0.6,
        "consistency": 0.5,
        "reliability": 0.4,
        "tradeoff-reasoning": 0.7,
    }

    record_system_design_attempt(db_session, activity_id=activity.id, scores=scores)

    high_score_mastery = get_mastery(db_session, "requirements")
    low_score_mastery = get_mastery(db_session, "reliability")
    assert high_score_mastery is not None
    assert low_score_mastery is not None
    assert high_score_mastery.mastery > low_score_mastery.mastery
    assert get_skill_events(db_session, "requirements")[0].source_type == "system_design_attempt"


def test_record_system_design_attempt_derives_overall_success_from_mean_score(db_session):
    activity = _create_system_design_activity(db_session)
    mostly_passing = {skill_id: 0.8 for skill_id in _TICKETMASTER_SKILLS}
    mostly_failing = {skill_id: 0.2 for skill_id in _TICKETMASTER_SKILLS}

    attempt_pass, _, _ = record_system_design_attempt(db_session, activity_id=activity.id, scores=mostly_passing)
    attempt_fail, _, _ = record_system_design_attempt(db_session, activity_id=activity.id, scores=mostly_failing)

    assert attempt_pass.success is True
    assert attempt_fail.success is False


def test_record_system_design_attempt_rejects_score_set_mismatch(db_session):
    activity = _create_system_design_activity(db_session)
    incomplete_scores = {"requirements": 0.9}

    with pytest.raises(ValueError, match="must cover exactly"):
        record_system_design_attempt(db_session, activity_id=activity.id, scores=incomplete_scores)


def test_record_system_design_attempt_rejects_non_system_design_activity(db_session):
    activity = create_activity(
        db_session,
        slug="two-sum",
        title="Two Sum",
        activity_type="coding",
        skill_ids=["hash-maps"],
    )

    with pytest.raises(ValueError, match="not a system design activity"):
        record_system_design_attempt(db_session, activity_id=activity.id, scores={"hash-maps": 0.8})


def test_get_system_design_attempt_round_trips(db_session):
    activity = _create_system_design_activity(db_session)
    scores = {skill_id: 0.6 for skill_id in _TICKETMASTER_SKILLS}
    attempt, result, _ = record_system_design_attempt(db_session, activity_id=activity.id, scores=scores)

    fetched = get_system_design_attempt(db_session, attempt.id)
    assert fetched is not None
    fetched_attempt, fetched_result = fetched
    assert fetched_attempt == attempt
    assert fetched_result == result


def test_get_system_design_attempt_returns_none_for_unknown_id(db_session):
    assert get_system_design_attempt(db_session, "does-not-exist") is None


def test_record_system_design_attempt_is_atomic_on_constraint_violation(db_session):
    activity = _create_system_design_activity(db_session)
    invalid_scores = {skill_id: 0.5 for skill_id in _TICKETMASTER_SKILLS}
    invalid_scores["requirements"] = 1.5  # out of the [0, 1] range enforced at the DB level

    with pytest.raises(IntegrityError):
        record_system_design_attempt(db_session, activity_id=activity.id, scores=invalid_scores)

    assert db_session.query(AttemptRecord).count() == 0
    assert db_session.query(SkillEventRecord).count() == 0
    assert get_mastery(db_session, "requirements") is None
