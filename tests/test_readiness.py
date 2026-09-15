from __future__ import annotations

import pytest

from swetrack.domains.learning.services import create_activity, record_attempt
from swetrack.domains.opportunities.models import CandidateProfile, JobRecord
from swetrack.domains.opportunities.readiness import compute_readiness, required_skill_ids
from swetrack.domains.opportunities.ranking.tfidf import TfidfRanker

_JOB = JobRecord(
    job_id="JOB-TEST",
    company="TestCo",
    title="Junior Backend Engineer",
    location="Remote",
    description="Build REST APIs in Python and Java.",
    skills=["Python", "Java", "REST APIs", "PostgreSQL", "Docker", "AWS Lambda", "Git"],
    experience_level="Entry-level",
    source="synthetic",
)

_PROFILE = CandidateProfile(
    skills=["Python", "Java", "AWS", "SQL"],
    experience="New grad backend engineer with Python and Java experience.",
    preferred_roles=["Backend Engineer"],
    preferred_locations=["Remote"],
)


def _record_many_successes(session, activity_id: str, count: int = 10) -> None:
    for _ in range(count):
        record_attempt(session, activity_id=activity_id, success=True)


def test_required_skill_ids_drops_unrecognized_free_text_terms():
    # Docker, AWS Lambda, and Git have no canonical taxonomy entry yet.
    assert set(required_skill_ids(_JOB)) == {"python", "java", "rest-apis", "sql"}


def test_compute_readiness_uses_default_mastery_for_unpracticed_skills(db_session):
    result = compute_readiness(db_session, job=_JOB, profile=_PROFILE, ranker=TfidfRanker())

    assert result.job_id == "JOB-TEST"
    assert len(result.skill_gaps) == 4
    assert all(gap.required == 1.0 for gap in result.skill_gaps)
    assert all(gap.mastery == pytest.approx(0.3) for gap in result.skill_gaps)  # default BKT p_init
    assert result.readiness_score == pytest.approx(0.3)


def test_compute_readiness_reflects_improved_mastery(db_session):
    activity = create_activity(
        db_session, slug="python-review", title="Python Review", activity_type="concept_review", skill_ids=["python"]
    )
    _record_many_successes(db_session, activity.id)

    result = compute_readiness(db_session, job=_JOB, profile=_PROFILE, ranker=TfidfRanker())

    python_gap = next(gap for gap in result.skill_gaps if gap.skill == "python")
    java_gap = next(gap for gap in result.skill_gaps if gap.skill == "java")
    assert python_gap.mastery > java_gap.mastery
    assert python_gap.gap < java_gap.gap


def test_compute_readiness_fit_score_is_independent_of_readiness_score(db_session):
    activity = create_activity(
        db_session, slug="python-review", title="Python Review", activity_type="concept_review", skill_ids=["python"]
    )
    _record_many_successes(db_session, activity.id)

    result = compute_readiness(db_session, job=_JOB, profile=_PROFILE, ranker=TfidfRanker())

    # Fit comes from candidate-profile <-> job text similarity, untouched by mastery.
    assert result.fit_score > 0.0
    assert result.readiness_score != result.fit_score


def test_compute_readiness_recommends_activities_relevant_to_required_skills(db_session):
    relevant = create_activity(
        db_session, slug="python-review", title="Python Review", activity_type="concept_review", skill_ids=["python"]
    )
    create_activity(
        db_session,
        slug="graph-practice",
        title="Graph Practice",
        activity_type="concept_review",
        skill_ids=["graphs"],
    )

    result = compute_readiness(db_session, job=_JOB, profile=_PROFILE, ranker=TfidfRanker())

    recommended_ids = {rec.activity.id for rec in result.recommended_activities}
    assert relevant.id in recommended_ids
    assert all("graphs" not in rec.activity.skill_ids for rec in result.recommended_activities)


def test_compute_readiness_handles_job_with_no_recognized_skills(db_session):
    unrecognized_job = JobRecord(
        job_id="JOB-UNRECOGNIZED",
        company="TestCo",
        title="Mystery Role",
        skills=["Quantum Computing", "Blockchain"],
        source="synthetic",
    )

    result = compute_readiness(db_session, job=unrecognized_job, profile=_PROFILE, ranker=TfidfRanker())

    assert result.skill_gaps == []
    assert result.readiness_score == 1.0
