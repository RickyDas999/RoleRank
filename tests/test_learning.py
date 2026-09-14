from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from swetrack.domains.learning.models import AttemptRecord
from swetrack.domains.learning.services import create_activity, get_skill_events, record_attempt


def test_create_activity_validates_skill_ids_against_taxonomy(db_session):
    with pytest.raises(ValueError, match="Unknown skill id"):
        create_activity(
            db_session,
            slug="graph-practice",
            title="Graph practice",
            activity_type="coding",
            skill_ids=["graphs", "not-a-real-skill"],
        )


def test_record_attempt_success_emits_one_skill_event_per_skill(db_session):
    activity = create_activity(
        db_session,
        slug="graph-practice",
        title="Graph practice",
        activity_type="coding",
        skill_ids=["graphs", "hash-maps"],
    )

    attempt, events = record_attempt(db_session, activity_id=activity.id, success=True)

    assert attempt.success is True
    assert {event.skill_id for event in events} == {"graphs", "hash-maps"}
    assert all(event.outcome == 1.0 for event in events)
    assert all(event.source_type == "coding_attempt" for event in events)
    assert all(event.source_id == attempt.id for event in events)


def test_record_attempt_failure_has_zero_outcome(db_session):
    activity = create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )

    _, events = record_attempt(db_session, activity_id=activity.id, success=False)

    assert len(events) == 1
    assert events[0].outcome == 0.0


def test_record_attempt_unknown_activity_raises(db_session):
    with pytest.raises(ValueError, match="Unknown learning activity id"):
        record_attempt(db_session, activity_id="does-not-exist", success=True)


def test_skill_events_are_replayable_in_chronological_order(db_session):
    activity = create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )

    record_attempt(db_session, activity_id=activity.id, success=False)
    record_attempt(db_session, activity_id=activity.id, success=True)
    record_attempt(db_session, activity_id=activity.id, success=True)

    history = get_skill_events(db_session, "dynamic-programming")

    assert [event.outcome for event in history] == [0.0, 1.0, 1.0]
    assert history == sorted(history, key=lambda event: event.timestamp)


def test_record_attempt_is_atomic_on_constraint_violation(db_session):
    activity = create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )

    with pytest.raises(IntegrityError):
        record_attempt(db_session, activity_id=activity.id, success=True, evidence_weight=0.0)

    # The Attempt must not be left orphaned by the failed SkillEvent insert.
    assert db_session.query(AttemptRecord).count() == 0
    assert get_skill_events(db_session, "dynamic-programming") == []
