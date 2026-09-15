"""SQLAlchemy ORM models backing the learning domain's persisted tables.

Only inserts are exposed by ``services.py`` for ``AttemptRecord`` and
``SkillEventRecord`` -- there is deliberately no update/delete path, since
both are meant to be immutable history. That immutability is a
service-layer discipline, not a database-level trigger, matching
CLAUDE.md's "avoid overengineering" guidance.

``SkillMasteryRecord`` is the one mutable table here: it is a *derived*,
recomputable cache of the current BKT mastery estimate per skill, kept in
sync with ``SkillEventRecord`` by ``services.record_attempt``. It is never
the source of truth -- the event history is -- and could always be rebuilt
by replaying ``SkillEventRecord`` rows from scratch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from swetrack.infrastructure.database.base import Base


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LearningActivityRecord(Base):
    __tablename__ = "learning_activities"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    slug: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(String)
    activity_type: Mapped[str] = mapped_column(String)
    skill_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[str | None] = mapped_column(String, nullable=True, default=None)


class AttemptRecord(Base):
    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    activity_id: Mapped[str] = mapped_column(String, ForeignKey("learning_activities.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    success: Mapped[bool] = mapped_column()
    notes: Mapped[str] = mapped_column(String, default="")


class SkillEventRecord(Base):
    __tablename__ = "skill_events"
    __table_args__ = (
        CheckConstraint("outcome >= 0.0 AND outcome <= 1.0", name="ck_skill_events_outcome_range"),
        CheckConstraint("evidence_weight > 0.0", name="ck_skill_events_evidence_weight_positive"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    skill_id: Mapped[str] = mapped_column(String, index=True)
    source_type: Mapped[str] = mapped_column(String)
    source_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    outcome: Mapped[float] = mapped_column(Float)
    evidence_weight: Mapped[float] = mapped_column(Float, default=1.0)


class CodingAttemptRecord(Base):
    """The coding-specific extension of one AttemptRecord (1:1, keyed by attempt_id).

    Kept separate from AttemptRecord rather than adding nullable columns to
    it, since these fields are only meaningful for coding attempts -- a
    System Design or concept-review attempt has no "hints used" or "mistake
    type." Phase 8's System Design rubric will get its own extension table
    the same way rather than reusing this one.
    """

    __tablename__ = "coding_attempt_details"
    __table_args__ = (
        CheckConstraint("hints_used >= 0", name="ck_coding_attempt_hints_used_nonnegative"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_coding_attempt_confidence_range",
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds > 0.0", name="ck_coding_attempt_duration_positive"
        ),
    )

    attempt_id: Mapped[str] = mapped_column(String, ForeignKey("attempts.id"), primary_key=True)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    mistake_type: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)


class SkillMasteryRecord(Base):
    __tablename__ = "skill_mastery"
    __table_args__ = (
        CheckConstraint("mastery >= 0.0 AND mastery <= 1.0", name="ck_skill_mastery_range"),
        CheckConstraint("event_count >= 0", name="ck_skill_mastery_event_count_nonnegative"),
    )

    skill_id: Mapped[str] = mapped_column(String, primary_key=True)
    mastery: Mapped[float] = mapped_column(Float)
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
