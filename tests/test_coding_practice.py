from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from swetrack.domains.learning.models import AttemptRecord, CodingAttemptRecord
from swetrack.domains.learning.services import (
    create_activity,
    get_coding_attempt,
    get_mastery,
    get_skill_events,
    record_coding_attempt,
)


def _create_coding_activity(session, **overrides):
    defaults = dict(
        slug="two-sum",
        title="Two Sum",
        activity_type="coding",
        skill_ids=["hash-maps"],
        difficulty="easy",
    )
    defaults.update(overrides)
    return create_activity(session, **defaults)


def test_record_coding_attempt_success_has_no_mistake_type(db_session):
    activity = _create_coding_activity(db_session)

    attempt, detail, events = record_coding_attempt(
        db_session,
        activity_id=activity.id,
        success=True,
        hints_used=1,
        confidence=0.8,
        duration_seconds=420.0,
    )

    assert attempt.success is True
    assert detail.attempt_id == attempt.id
    assert detail.hints_used == 1
    assert detail.confidence == 0.8
    assert detail.duration_seconds == 420.0
    assert detail.mistake_type is None
    assert {event.skill_id for event in events} == {"hash-maps"}


def test_record_coding_attempt_failure_with_mistake_type(db_session):
    activity = _create_coding_activity(db_session)

    attempt, detail, _ = record_coding_attempt(
        db_session,
        activity_id=activity.id,
        success=False,
        mistake_type="edge_case",
    )

    assert attempt.success is False
    assert detail.mistake_type == "edge_case"


def test_record_coding_attempt_rejects_mistake_type_on_success(db_session):
    activity = _create_coding_activity(db_session)

    with pytest.raises(ValueError, match="mistake_type must be None"):
        record_coding_attempt(
            db_session,
            activity_id=activity.id,
            success=True,
            mistake_type="off_by_one",
        )


def test_record_coding_attempt_rejects_non_coding_activity(db_session):
    activity = create_activity(
        db_session,
        slug="dp-review",
        title="DP concept review",
        activity_type="concept_review",
        skill_ids=["dynamic-programming"],
    )

    with pytest.raises(ValueError, match="not a coding activity"):
        record_coding_attempt(db_session, activity_id=activity.id, success=True)


def test_record_coding_attempt_updates_mastery_like_generic_attempt(db_session):
    activity = _create_coding_activity(db_session)

    record_coding_attempt(db_session, activity_id=activity.id, success=True)

    mastery = get_mastery(db_session, "hash-maps")
    assert mastery is not None
    assert mastery.event_count == 1
    assert get_skill_events(db_session, "hash-maps")[0].source_type == "coding_attempt"


def test_get_coding_attempt_round_trips(db_session):
    activity = _create_coding_activity(db_session)
    attempt, detail, _ = record_coding_attempt(
        db_session, activity_id=activity.id, success=False, hints_used=2, mistake_type="recurrence"
    )

    result = get_coding_attempt(db_session, attempt.id)
    assert result is not None
    fetched_attempt, fetched_detail = result
    assert fetched_attempt == attempt
    assert fetched_detail == detail


def test_get_coding_attempt_returns_none_for_unknown_id(db_session):
    assert get_coding_attempt(db_session, "does-not-exist") is None


def test_record_coding_attempt_is_atomic_on_constraint_violation(db_session):
    activity = _create_coding_activity(db_session)

    with pytest.raises(IntegrityError):
        record_coding_attempt(db_session, activity_id=activity.id, success=True, hints_used=-1)

    assert db_session.query(AttemptRecord).count() == 0
    assert db_session.query(CodingAttemptRecord).count() == 0
    assert get_mastery(db_session, "hash-maps") is None


def test_activity_difficulty_round_trips(db_session):
    activity = _create_coding_activity(db_session, slug="hard-problem", difficulty="hard")
    assert activity.difficulty == "hard"
