"""Bayesian Knowledge Tracing (BKT): a configurable, deterministic mastery model.

Pure functions only -- no database or business-logic imports here, so this
module is unit-testable in complete isolation (CLAUDE.md Phase 9: "The
implementation should live outside business/database logic").

Standard two-state HMM formulation (Corbett & Anderson, 1994):

    P(L0) -- prior probability the skill is already known
    P(T)  -- probability of transitioning from unknown to known after one
             practice opportunity
    P(G)  -- probability of answering correctly while NOT knowing the skill
    P(S)  -- probability of answering incorrectly while knowing the skill

``DEFAULT_PARAMETERS`` below are commonly used illustrative values, not fit
to this project's data -- they are configured defaults, not learned ones
(CLAUDE.md: "Do not pretend initial BKT parameter values were learned if
they are configured defaults"). See ``ml/evaluation/knowledge_tracing.py``
for an honest comparison against a naive historical-success-rate baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

_EPSILON = 1e-9


class BKTParameters(BaseModel):
    """Configurable BKT parameters. All four are probabilities in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    p_init: float = Field(ge=0.0, le=1.0, description="P(L0): prior probability of already knowing the skill")
    p_transit: float = Field(
        ge=0.0, le=1.0, description="P(T): probability of learning the skill on one opportunity"
    )
    p_guess: float = Field(
        ge=0.0, le=1.0, description="P(G): probability of a correct answer while not knowing the skill"
    )
    p_slip: float = Field(
        ge=0.0, le=1.0, description="P(S): probability of an incorrect answer while knowing the skill"
    )


DEFAULT_PARAMETERS = BKTParameters(p_init=0.3, p_transit=0.1, p_guess=0.2, p_slip=0.1)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def update_mastery(prior_mastery: float, correct: bool, params: BKTParameters) -> float:
    """One BKT step: a Bayesian posterior update on ``correct``, then the learning transition.

    Numerically safe: if the posterior's denominator is ~0 (only possible at
    degenerate parameter/prior combinations, e.g. ``prior_mastery=0`` with
    ``p_guess=0``), falls back to the un-updated prior rather than raising
    ``ZeroDivisionError``.
    """
    if correct:
        numerator = prior_mastery * (1.0 - params.p_slip)
        denominator = numerator + (1.0 - prior_mastery) * params.p_guess
    else:
        numerator = prior_mastery * params.p_slip
        denominator = numerator + (1.0 - prior_mastery) * (1.0 - params.p_guess)

    posterior = numerator / denominator if denominator > _EPSILON else prior_mastery
    mastery_after_transition = posterior + (1.0 - posterior) * params.p_transit
    return _clamp(mastery_after_transition)


def trace_mastery(outcomes: Sequence[bool], params: BKTParameters, *, p_init: float | None = None) -> list[float]:
    """Replay a chronological sequence of correct/incorrect outcomes.

    Returns the mastery estimate *after* each outcome (same length as
    ``outcomes``). Deterministic: identical inputs always produce identical
    output.
    """
    mastery = _clamp(p_init) if p_init is not None else params.p_init
    history: list[float] = []
    for correct in outcomes:
        mastery = update_mastery(mastery, correct, params)
        history.append(mastery)
    return history


def predicted_correct_probability(mastery: float, params: BKTParameters) -> float:
    """P(correct on the next opportunity) given a current mastery estimate."""
    return mastery * (1.0 - params.p_slip) + (1.0 - mastery) * params.p_guess
