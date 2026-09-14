"""Learning domain service layer: the only way callers write to this domain.

``record_attempt`` is the important one -- it writes one Attempt, one
SkillEvent per skill the activity exercises, and one sequential BKT mastery
update per skill, all as a single transaction. Either all of it commits, or
none of it does (see test_learning.py for a forced mid-transaction failure
that verifies the rollback).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from swetrack.domains.learning.models import (
    AttemptRecord,
    LearningActivityRecord,
    SkillEventRecord,
    SkillMasteryRecord,
)
from swetrack.domains.learning.schemas import (
    ActivityType,
    Attempt,
    LearningActivity,
    SkillEvent,
    SkillEventSourceType,
    SkillMastery,
)
from swetrack.domains.skills.normalization import get_skill_by_id
from swetrack.ml.knowledge_tracing.bkt import DEFAULT_PARAMETERS, BKTParameters, update_mastery

_SOURCE_TYPE_BY_ACTIVITY_TYPE: dict[ActivityType, SkillEventSourceType] = {
    "coding": "coding_attempt",
    "system_design": "system_design_attempt",
    "concept_review": "concept_review",
}


def create_activity(
    session: Session,
    *,
    slug: str,
    title: str,
    activity_type: ActivityType,
    skill_ids: list[str],
) -> LearningActivity:
    """Create a practiceable activity, validating every skill_id against the canonical taxonomy."""
    unknown = [skill_id for skill_id in skill_ids if get_skill_by_id(skill_id) is None]
    if unknown:
        raise ValueError(f"Unknown skill id(s): {unknown}")

    record = LearningActivityRecord(
        slug=slug, title=title, activity_type=activity_type, skill_ids=list(skill_ids)
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _to_activity(record)


def record_attempt(
    session: Session,
    *,
    activity_id: str,
    success: bool,
    notes: str = "",
    evidence_weight: float = 1.0,
    bkt_params: BKTParameters = DEFAULT_PARAMETERS,
) -> tuple[Attempt, list[SkillEvent]]:
    """Record one attempt, its SkillEvents, and each skill's updated BKT mastery.

    Mastery is updated *sequentially*: one BKT step is applied to the cached
    prior mastery (or ``bkt_params.p_init`` if this is the skill's first
    observation), not a full replay of history -- see
    ``ml/knowledge_tracing/bkt.py``.

    Raises ``ValueError`` if the activity does not exist. Raises and rolls
    back if any SkillEvent or mastery row violates its DB-level constraints
    (e.g. a non-positive ``evidence_weight``) -- neither the Attempt nor any
    partial mastery update is left behind.
    """
    activity = session.get(LearningActivityRecord, activity_id)
    if activity is None:
        raise ValueError(f"Unknown learning activity id: {activity_id!r}")

    source_type = _SOURCE_TYPE_BY_ACTIVITY_TYPE[activity.activity_type]  # type: ignore[index]
    outcome = 1.0 if success else 0.0

    try:
        attempt_record = AttemptRecord(activity_id=activity_id, success=success, notes=notes)
        session.add(attempt_record)
        session.flush()  # assign attempt_record.id without committing yet

        event_records = [
            SkillEventRecord(
                skill_id=skill_id,
                source_type=source_type,
                source_id=attempt_record.id,
                outcome=outcome,
                evidence_weight=evidence_weight,
            )
            for skill_id in activity.skill_ids
        ]
        session.add_all(event_records)
        session.flush()  # surface any SkillEvent constraint violation before touching mastery

        for skill_id in activity.skill_ids:
            _apply_sequential_mastery_update(session, skill_id, correct=success, params=bkt_params)

        session.commit()
    except Exception:
        session.rollback()
        raise

    session.refresh(attempt_record)
    for record in event_records:
        session.refresh(record)
    return _to_attempt(attempt_record), [_to_skill_event(record) for record in event_records]


def get_skill_events(session: Session, skill_id: str) -> list[SkillEvent]:
    """Return a skill's full immutable event history in chronological order."""
    records = (
        session.query(SkillEventRecord)
        .filter(SkillEventRecord.skill_id == skill_id)
        .order_by(SkillEventRecord.timestamp.asc())
        .all()
    )
    return [_to_skill_event(record) for record in records]


def get_mastery(session: Session, skill_id: str) -> SkillMastery | None:
    """Return the cached BKT mastery estimate for a skill, or None if it has no history yet."""
    record = session.get(SkillMasteryRecord, skill_id)
    return _to_mastery(record) if record is not None else None


def _apply_sequential_mastery_update(
    session: Session, skill_id: str, *, correct: bool, params: BKTParameters
) -> SkillMasteryRecord:
    """Apply one BKT step to the cached mastery for a skill, creating it if absent."""
    record = session.get(SkillMasteryRecord, skill_id)
    prior_mastery = record.mastery if record is not None else params.p_init
    new_mastery = update_mastery(prior_mastery, correct, params)

    if record is None:
        record = SkillMasteryRecord(skill_id=skill_id, mastery=new_mastery, event_count=1)
        session.add(record)
    else:
        record.mastery = new_mastery
        record.event_count += 1
    return record


def _to_activity(record: LearningActivityRecord) -> LearningActivity:
    return LearningActivity(
        id=record.id,
        slug=record.slug,
        title=record.title,
        activity_type=record.activity_type,  # type: ignore[arg-type]
        skill_ids=list(record.skill_ids),
    )


def _to_attempt(record: AttemptRecord) -> Attempt:
    return Attempt(
        id=record.id,
        activity_id=record.activity_id,
        timestamp=record.timestamp,
        success=record.success,
        notes=record.notes,
    )


def _to_skill_event(record: SkillEventRecord) -> SkillEvent:
    return SkillEvent(
        id=record.id,
        skill_id=record.skill_id,
        source_type=record.source_type,  # type: ignore[arg-type]
        source_id=record.source_id,
        timestamp=record.timestamp,
        outcome=record.outcome,
        evidence_weight=record.evidence_weight,
    )


def _to_mastery(record: SkillMasteryRecord) -> SkillMastery:
    return SkillMastery(
        skill_id=record.skill_id,
        mastery=record.mastery,
        event_count=record.event_count,
        updated_at=record.updated_at,
    )
