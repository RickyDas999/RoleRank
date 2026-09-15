"""Explainable heuristic study ranker (CLAUDE.md Phase 10).

Pure functions only -- no database or business-logic imports here, mirroring
``ml/knowledge_tracing/bkt.py``'s separation. This ranker is a deterministic
heuristic, not a learned model: only the mastery estimates it consumes as
input come from actual ML (BKT). Job/interview demand components from
CLAUDE.md's full component list are deferred to M10, which is what connects
opportunity skill requirements to this domain -- there is no such signal to
rank against yet.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

# How well a difficulty level suits a given mastery gap: "easy" is the best
# fit when a skill is almost entirely unmastered, "hard" when it is nearly
# mastered, so recommendations grow harder as the learner improves.
_IDEAL_GAP_BY_DIFFICULTY: dict[str, float] = {"easy": 0.8, "medium": 0.5, "hard": 0.2}

# Days of no practice before staleness saturates towards 1.0. A configured
# constant, not fit to any data -- see bkt.py's DEFAULT_PARAMETERS docstring
# for the same "configured, not learned" distinction.
_DEFAULT_STALENESS_HALF_LIFE_DAYS = 14.0

# Recent-attempt count at which repetition_penalty saturates to 1.0.
_DEFAULT_REPETITION_CAP = 5


class StudyRankingWeights(BaseModel):
    """Configurable weights combining the four ranking components into one score.

    ``mastery_gap + staleness + difficulty_fit`` should sum to 1.0 so the
    pre-penalty score stays in [0, 1]; ``repetition_penalty`` is a separate
    subtractive weight, not part of that sum.
    """

    model_config = ConfigDict(frozen=True)

    mastery_gap: float = Field(ge=0.0, le=1.0, default=0.45)
    staleness: float = Field(ge=0.0, le=1.0, default=0.30)
    difficulty_fit: float = Field(ge=0.0, le=1.0, default=0.25)
    repetition_penalty: float = Field(ge=0.0, le=1.0, default=0.2)


DEFAULT_WEIGHTS = StudyRankingWeights()


class StudyRankingComponents(BaseModel):
    """The per-component breakdown behind one activity's ranking score."""

    model_config = ConfigDict(frozen=True)

    mastery_gap: float = Field(ge=0.0, le=1.0)
    staleness: float = Field(ge=0.0, le=1.0)
    difficulty_fit: float = Field(ge=0.0, le=1.0)
    repetition_penalty: float = Field(ge=0.0, le=1.0)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def compute_mastery_gap(mastery: float) -> float:
    """How far a skill is from full mastery: 1.0 (unmastered) down to 0.0 (mastered)."""
    return _clamp(1.0 - mastery)


def compute_staleness(days_since_last_practice: float | None, *, half_life_days: float = _DEFAULT_STALENESS_HALF_LIFE_DAYS) -> float:
    """How overdue a skill is for review: 0.0 (just practiced) approaching 1.0 (long overdue).

    A skill with no recorded practice at all (``None``) is treated as
    maximally stale, since there is nothing to prioritize over it.
    """
    if days_since_last_practice is None:
        return 1.0
    return _clamp(1.0 - math.exp(-max(0.0, days_since_last_practice) / half_life_days))


def compute_difficulty_fit(mastery_gap: float, difficulty: str | None) -> float:
    """How well an activity's declared difficulty suits the learner's current mastery gap.

    An activity with no declared difficulty gets a neutral 0.5 -- there is
    no signal to score a fit against.
    """
    ideal_gap = _IDEAL_GAP_BY_DIFFICULTY.get(difficulty) if difficulty else None
    if ideal_gap is None:
        return 0.5
    return _clamp(1.0 - abs(mastery_gap - ideal_gap))


def compute_repetition_penalty(recent_attempt_count: int, *, cap: int = _DEFAULT_REPETITION_CAP) -> float:
    """How much to deprioritize an activity that has already been drilled recently."""
    if recent_attempt_count <= 0:
        return 0.0
    return _clamp(recent_attempt_count / cap)


def score_activity(components: StudyRankingComponents, weights: StudyRankingWeights = DEFAULT_WEIGHTS) -> float:
    """Combine one activity's components into a single explainable score."""
    raw = (
        weights.mastery_gap * components.mastery_gap
        + weights.staleness * components.staleness
        + weights.difficulty_fit * components.difficulty_fit
        - weights.repetition_penalty * components.repetition_penalty
    )
    return _clamp(raw)
