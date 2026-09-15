"""Validated, immutable read schemas for the learning domain.

These are decoupled from the SQLAlchemy ORM models in ``models.py`` so
callers never hold a session-bound ORM instance -- they get back a frozen
Pydantic snapshot instead. System-Design-specific attempt fields (rubric
scores) are deferred to M8 rather than speculatively added here.
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
Difficulty = Literal["easy", "medium", "hard"]
MistakeType = Literal[
    "algorithm_selection",
    "state_definition",
    "recurrence",
    "off_by_one",
    "pointer_management",
    "base_case",
    "complexity",
    "data_structure_choice",
    "mutation",
    "edge_case",
]


class LearningActivity(BaseModel):
    """A practiceable activity (e.g. 'Graph practice') tied to one or more skills."""

    model_config = ConfigDict(frozen=True)

    id: str
    slug: str
    title: str
    activity_type: ActivityType
    skill_ids: list[str] = Field(default_factory=list)
    difficulty: Difficulty | None = None


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


class CodingAttemptDetail(BaseModel):
    """The coding-specific extension of one Attempt (CLAUDE.md Phase 7).

    ``mistake_type`` is only meaningful for a failed attempt -- an attempt
    that succeeded has nothing to categorize as a mistake. Validated in
    ``services.record_coding_attempt``, not here, since it depends on the
    sibling Attempt's ``success`` value.
    """

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    hints_used: int = Field(ge=0, default=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mistake_type: MistakeType | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)


class SkillMastery(BaseModel):
    """The current cached BKT mastery estimate for one skill.

    Derived, not authoritative: always recomputable from that skill's full
    SkillEvent history (see ``services.get_skill_events``). ``event_count``
    records how many events have been folded into ``mastery`` so far.
    """

    model_config = ConfigDict(frozen=True)

    skill_id: str
    mastery: float = Field(ge=0.0, le=1.0)
    event_count: int = Field(ge=0)
    updated_at: datetime
