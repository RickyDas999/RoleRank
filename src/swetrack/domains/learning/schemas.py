"""Validated, immutable read schemas for the learning domain.

These are decoupled from the SQLAlchemy ORM models in ``models.py`` so
callers never hold a session-bound ORM instance -- they get back a frozen
Pydantic snapshot instead. Richer, activity-type-specific attempt fields
(hints used, mistake category, System Design rubric scores) are deferred to
the milestones that actually need them (M7 coding practice, M8 System
Design practice) rather than speculatively added here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ActivityType = Literal["coding", "system_design", "concept_review"]
SkillEventSourceType = Literal[
    "coding_attempt",
    "system_design_attempt",
    "interview_feedback",
    "concept_review",
]


class LearningActivity(BaseModel):
    """A practiceable activity (e.g. 'Graph practice') tied to one or more skills."""

    model_config = ConfigDict(frozen=True)

    id: str
    slug: str
    title: str
    activity_type: ActivityType
    skill_ids: list[str] = Field(default_factory=list)


class Attempt(BaseModel):
    """One practice attempt against a LearningActivity."""

    model_config = ConfigDict(frozen=True)

    id: str
    activity_id: str
    timestamp: datetime
    success: bool
    notes: str = ""


class SkillEvent(BaseModel):
    """One immutable observation of evidence about a skill.

    ``outcome`` is a continuous strength-of-evidence signal in [0, 1] rather
    than a plain success/failure bit, so future sources (e.g. a System
    Design rubric score) can report partial credit without changing this
    schema. ``evidence_weight`` lets some observations count more than
    others (e.g. a harder problem, or more confident interview feedback).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    skill_id: str
    source_type: SkillEventSourceType
    source_id: str
    timestamp: datetime
    outcome: float = Field(ge=0.0, le=1.0)
    evidence_weight: float = Field(gt=0.0, default=1.0)
