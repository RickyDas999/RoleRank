"""Chronological baseline evaluation for the BKT mastery model.

Compares BKT's next-opportunity correctness prediction against the naive
"historical success rate so far" baseline, per CLAUDE.md's Evaluation
Philosophy for knowledge tracing: compare against historical success rate
per skill, score with Brier score and log loss, and evaluate
chronologically -- never with a random train/test split of sequential
learner data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from sklearn.metrics import brier_score_loss, log_loss

from swetrack.ml.knowledge_tracing.bkt import BKTParameters, predicted_correct_probability, trace_mastery


class _ScorePair(TypedDict):
    brier_score: float
    log_loss: float


class WalkForwardResult(TypedDict):
    n_evaluated: int
    bkt: _ScorePair
    baseline_historical_success_rate: _ScorePair


def walk_forward_evaluate(outcomes: Sequence[bool], params: BKTParameters) -> WalkForwardResult:
    """Chronologically evaluate BKT vs. a naive historical-success-rate baseline.

    For each opportunity ``t`` (starting at 1), both models predict
    P(correct) using only ``outcomes[:t]`` -- never a later observation --
    then are scored against the actual ``outcomes[t]``. This is a
    walk-forward evaluation, the sequential-data equivalent of a held-out
    test set; it is not accuracy on training data.
    """
    if len(outcomes) < 2:
        raise ValueError("Need at least 2 outcomes to evaluate one chronological prediction")

    actuals: list[int] = []
    bkt_predictions: list[float] = []
    baseline_predictions: list[float] = []

    for t in range(1, len(outcomes)):
        history = outcomes[:t]
        mastery = trace_mastery(history, params)[-1]
        bkt_predictions.append(predicted_correct_probability(mastery, params))
        baseline_predictions.append(sum(history) / len(history))
        actuals.append(1 if outcomes[t] else 0)

    return {
        "n_evaluated": len(actuals),
        "bkt": {
            "brier_score": brier_score_loss(actuals, bkt_predictions),
            "log_loss": log_loss(actuals, bkt_predictions, labels=[0, 1]),
        },
        "baseline_historical_success_rate": {
            "brier_score": brier_score_loss(actuals, baseline_predictions),
            "log_loss": log_loss(actuals, baseline_predictions, labels=[0, 1]),
        },
    }
