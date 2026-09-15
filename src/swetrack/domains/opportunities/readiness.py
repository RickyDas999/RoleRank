"""Role readiness: connects opportunity skill requirements to learner mastery.

CLAUDE.md Phase 11's flagship integration. Deliberately distinct from Role
Fit (semantic candidate<->job similarity, see ``ranking/``): Fit answers "how
well does this opportunity align with the candidate," Readiness answers "how
prepared is the user for the skills this opportunity requires." CLAUDE.md:
"Never conflate them."
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from swetrack.domains.learning.schemas import StudyRecommendation
from swetrack.domains.learning.services import get_mastery, get_study_recommendations
from swetrack.domains.opportunities.models import CandidateProfile, JobRecord
from swetrack.domains.opportunities.ranking.base import Ranker
from swetrack.domains.skills.normalization import normalize_skill
from swetrack.ml.knowledge_tracing.bkt import DEFAULT_PARAMETERS, BKTParameters

_RECOMMENDED_ACTIVITY_LIMIT = 3


class SkillGap(BaseModel):
    """One required skill's gap between what a job needs and the learner's current mastery."""

    model_config = ConfigDict(frozen=True)

    skill: str
    required: float = Field(ge=0.0, le=1.0)
    mastery: float = Field(ge=0.0, le=1.0)
    gap: float = Field(ge=0.0, le=1.0)


class ReadinessResult(BaseModel):
    """Role Fit and Readiness for one job, kept as two separate scores (CLAUDE.md Phase 11)."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    fit_score: float
    readiness_score: float = Field(ge=0.0, le=1.0)
    skill_gaps: list[SkillGap]
    recommended_activities: list[StudyRecommendation]


def required_skill_ids(job: JobRecord) -> list[str]:
    """Canonical skill ids a job requires, deduplicated, dropping unrecognized free-text terms.

    Every recognized skill is treated as required at full importance (1.0):
    job postings carry no explicit per-skill weighting/emphasis signal today
    (CLAUDE.md Phase 5 mapped free-text skills to the canonical taxonomy, but
    never assigned per-skill importance), so inventing one here would be a
    fabricated precision CLAUDE.md explicitly warns against ("honest
    metrics... no fabricated claims").
    """
    seen: dict[str, None] = {}
    for raw_skill in job.skills:
        skill = normalize_skill(raw_skill)
        if skill is not None:
            seen.setdefault(skill.id, None)
    return list(seen.keys())


def compute_readiness(
    session: Session,
    *,
    job: JobRecord,
    profile: CandidateProfile,
    ranker: Ranker,
    bkt_params: BKTParameters = DEFAULT_PARAMETERS,
    recommended_activity_limit: int = _RECOMMENDED_ACTIVITY_LIMIT,
) -> ReadinessResult:
    """Compute Role Fit and Readiness for one job against the candidate profile and tracked mastery.

    Fit comes entirely from the existing semantic ranker (candidate profile
    <-> job text); Readiness comes entirely from BKT-tracked skill mastery
    against the job's canonical required skills. Two independent signals
    computed from unrelated inputs, never blended into one number.

    A skill with no recorded practice history uses ``bkt_params.p_init`` as
    its mastery, matching how ``get_study_recommendations`` (M9) treats an
    unpracticed skill -- consistent "no history yet" default across both.
    If the job has no recognized required skills, ``readiness_score`` is
    1.0 (nothing to be unprepared for) and recommendations fall back to the
    unfiltered top study activities rather than an empty list.
    """
    fit_score = ranker.score_jobs(profile, [job])[0].score

    skill_ids = required_skill_ids(job)
    skill_gaps: list[SkillGap] = []
    for skill_id in skill_ids:
        mastery_record = get_mastery(session, skill_id)
        mastery = mastery_record.mastery if mastery_record is not None else bkt_params.p_init
        skill_gaps.append(SkillGap(skill=skill_id, required=1.0, mastery=mastery, gap=1.0 - mastery))

    readiness_score = sum(gap.mastery for gap in skill_gaps) / len(skill_gaps) if skill_gaps else 1.0

    recommended_activities = get_study_recommendations(
        session,
        top_k=recommended_activity_limit,
        skill_ids=set(skill_ids) or None,
        bkt_params=bkt_params,
    )

    return ReadinessResult(
        job_id=job.job_id,
        fit_score=fit_score,
        readiness_score=readiness_score,
        skill_gaps=skill_gaps,
        recommended_activities=recommended_activities,
    )
