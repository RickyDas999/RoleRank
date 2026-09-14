"""SQLAlchemy ORM models backing the learning domain's persisted tables.

Only inserts are exposed by ``services.py`` -- there is deliberately no
update/delete path for ``AttemptRecord`` or ``SkillEventRecord``, since both
are meant to be immutable history. That immutability is a service-layer
discipline, not a database-level trigger, matching CLAUDE.md's "avoid
overengineering" guidance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, String
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
