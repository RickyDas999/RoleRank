from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from swetrack.domains.learning.models import AttemptRecord
from swetrack.domains.learning.services import create_activity, get_mastery, get_skill_events, record_attempt
from swetrack.infrastructure.database.base import get_engine, get_sessionmaker, init_db
from swetrack.ml.knowledge_tracing.bkt import DEFAULT_PARAMETERS, trace_mastery


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

    # The Attempt must not be left orphaned by the failed SkillEvent insert,
    # and no partial mastery update should have been applied either.
    assert db_session.query(AttemptRecord).count() == 0
    assert get_skill_events(db_session, "dynamic-programming") == []
    assert get_mastery(db_session, "dynamic-programming") is None


def test_record_attempt_creates_mastery_matching_default_p_init_on_first_event(db_session):
    activity = create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )

    record_attempt(db_session, activity_id=activity.id, success=True)

    mastery = get_mastery(db_session, "dynamic-programming")
    assert mastery is not None
    assert mastery.event_count == 1
    expected = trace_mastery([True], DEFAULT_PARAMETERS)[-1]
    assert mastery.mastery == pytest.approx(expected)


def test_sequential_mastery_updates_match_a_full_replay(db_session):
    activity = create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )
    outcomes = [False, True, True, False, True]

    for success in outcomes:
        record_attempt(db_session, activity_id=activity.id, success=success)

    mastery = get_mastery(db_session, "dynamic-programming")
    assert mastery is not None
    assert mastery.event_count == len(outcomes)

    expected = trace_mastery(outcomes, DEFAULT_PARAMETERS)[-1]
    assert mastery.mastery == pytest.approx(expected)


def test_get_mastery_returns_none_with_no_history(db_session):
    create_activity(
        db_session, slug="dp-practice", title="DP practice", activity_type="coding", skill_ids=["dynamic-programming"]
    )
    assert get_mastery(db_session, "dynamic-programming") is None


def test_mastery_persists_across_sessions_on_the_same_database():
    # Uses its own engine (rather than the db_session fixture) to prove
    # mastery survives closing and reopening a session against the same
    # underlying database, not just within one in-memory session.
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    Session = get_sessionmaker(engine)

    write_session = Session()
    activity = create_activity(
        write_session,
        slug="dp-practice",
        title="DP practice",
        activity_type="coding",
        skill_ids=["dynamic-programming"],
    )
    record_attempt(write_session, activity_id=activity.id, success=True)
    write_session.close()

    read_session = Session()
    mastery = get_mastery(read_session, "dynamic-programming")
    read_session.close()
    engine.dispose()

    assert mastery is not None
    assert mastery.event_count == 1
