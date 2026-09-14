"""Run the BKT-vs-baseline knowledge tracing evaluation on synthetic data.

SYNTHETIC DEMONSTRATION DATA ONLY. There is no real multi-attempt learner
history yet (the learning domain was only just introduced in Milestone 5),
so this script simulates one deterministically-seeded learner practicing
one skill, using a *true* generative BKT process, and evaluates the
model's DEFAULT_PARAMETERS (deliberately not fit to the simulation's true
parameters) against a naive historical-success-rate baseline. Do not treat
these results as evidence about real users -- they only demonstrate that
the evaluation harness is wired correctly (CLAUDE.md Data Integrity:
"Never describe synthetic history as real usage").
"""

from __future__ import annotations

import argparse
import random

from swetrack.ml.evaluation.knowledge_tracing import walk_forward_evaluate
from swetrack.ml.knowledge_tracing.bkt import DEFAULT_PARAMETERS, BKTParameters

# Deliberately different from DEFAULT_PARAMETERS, so the evaluated model is
# genuinely misspecified relative to the data-generating process -- the
# same situation a real deployment would face with unfit default parameters.
_TRUE_SIMULATION_PARAMETERS = BKTParameters(p_init=0.2, p_transit=0.15, p_guess=0.15, p_slip=0.1)


def simulate_learner_outcomes(n_opportunities: int, seed: int, params: BKTParameters) -> list[bool]:
    """Deterministically simulate one learner's correct/incorrect sequence.

    Tracks a hidden "knows the skill" state that starts at ``params.p_init``
    and may transition to True (probability ``params.p_transit``) after each
    opportunity. Each observed outcome is correct with probability
    ``1 - p_slip`` if the hidden state is True, or ``p_guess`` if False.
    """
    rng = random.Random(seed)
    knows_skill = rng.random() < params.p_init
    outcomes: list[bool] = []
    for _ in range(n_opportunities):
        p_correct = (1.0 - params.p_slip) if knows_skill else params.p_guess
        outcomes.append(rng.random() < p_correct)
        if not knows_skill and rng.random() < params.p_transit:
            knows_skill = True
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-opportunities", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outcomes = simulate_learner_outcomes(args.n_opportunities, args.seed, _TRUE_SIMULATION_PARAMETERS)
    result = walk_forward_evaluate(outcomes, DEFAULT_PARAMETERS)

    print("SYNTHETIC DEMONSTRATION DATA -- not real learner history.")
    print(f"Simulated {args.n_opportunities} opportunities (seed={args.seed}): {outcomes}")
    print(f"Evaluated {result['n_evaluated']} chronological predictions.\n")

    print(f"{'Model':<32} {'Brier score':>12} {'Log loss':>12}")
    print(f"{'BKT (default parameters)':<32} {result['bkt']['brier_score']:>12.4f} {result['bkt']['log_loss']:>12.4f}")
    baseline = result["baseline_historical_success_rate"]
    print(f"{'Historical success rate':<32} {baseline['brier_score']:>12.4f} {baseline['log_loss']:>12.4f}")

    bkt_brier = result["bkt"]["brier_score"]
    baseline_brier = baseline["brier_score"]
    if bkt_brier < baseline_brier:
        print("\nBKT scored lower Brier loss than the naive baseline on this run.")
    elif bkt_brier > baseline_brier:
        print("\nThe naive baseline scored lower Brier loss than BKT on this run.")
    else:
        print("\nBKT and the naive baseline tied on Brier loss on this run.")
    print(
        "This is one synthetic run with untuned default parameters, not a claim "
        "that BKT generalizes to real learners."
    )


if __name__ == "__main__":
    main()
