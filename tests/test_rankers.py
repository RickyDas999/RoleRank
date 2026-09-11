"""Tests for the shared Ranker interface and TfidfRanker."""

from __future__ import annotations

import pytest

from rolerank.models import CandidateProfile, JobRecord
from rolerank.ranking.tfidf import TfidfRanker


def _profile(**overrides: object) -> CandidateProfile:
    defaults: dict[str, object] = dict(
        skills=["Python", "AWS", "FastAPI"],
        experience="Backend engineer building REST APIs in Python and deploying to AWS.",
        preferred_roles=["Backend Engineer"],
        preferred_locations=["Remote"],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(job_id: str, **overrides: object) -> JobRecord:
    defaults: dict[str, object] = dict(
        job_id=job_id,
        company="Acme",
        title="Backend Engineer",
        location="Remote",
        description="Build REST APIs in Python and deploy to AWS.",
        skills=["Python", "AWS", "FastAPI"],
        experience_level="Entry-level",
        source="synthetic",
    )
    defaults.update(overrides)
    return JobRecord(**defaults)


def test_recommend_returns_exactly_k_unique_jobs():
    profile = _profile()
    jobs = [_job(f"JOB-{i}") for i in range(1, 6)]
    results = TfidfRanker().recommend(profile, jobs, top_k=3)
    assert len(results) == 3
    assert len({r.job.job_id for r in results}) == 3


def test_scores_are_non_increasing():
    profile = _profile()
    jobs = [_job(f"JOB-{i}") for i in range(1, 6)]
    results = TfidfRanker().recommend(profile, jobs, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ties_broken_by_ascending_job_id():
    profile = _profile()
    # Identical job text yields identical scores, so ordering must fall back to job_id.
    jobs = [_job("JOB-B"), _job("JOB-A"), _job("JOB-C")]
    results = TfidfRanker().recommend(profile, jobs, top_k=3)
    assert [r.job.job_id for r in results] == ["JOB-A", "JOB-B", "JOB-C"]


def test_strong_exact_match_outranks_unrelated_job():
    profile = _profile(
        skills=["Python", "FastAPI", "AWS Lambda"],
        experience="Python FastAPI backend engineer deploying AWS Lambda REST APIs.",
    )
    strong_match = _job(
        "JOB-MATCH",
        title="Python FastAPI Backend Engineer",
        description="Python FastAPI backend engineer deploying AWS Lambda REST APIs.",
        skills=["Python", "FastAPI", "AWS Lambda"],
    )
    unrelated = _job(
        "JOB-UNRELATED",
        title="Graphic Designer",
        description="Create marketing graphics and brand illustrations using Adobe Photoshop.",
        skills=["Adobe Photoshop", "Illustrator", "Branding"],
    )
    results = TfidfRanker().recommend(profile, [strong_match, unrelated], top_k=2)
    assert results[0].job.job_id == "JOB-MATCH"
    assert results[0].score > results[1].score


def test_recommend_rejects_top_k_greater_than_available_jobs():
    profile = _profile()
    jobs = [_job("JOB-1")]
    with pytest.raises(ValueError):
        TfidfRanker().recommend(profile, jobs, top_k=2)


def test_recommend_rejects_empty_job_list():
    profile = _profile()
    with pytest.raises(ValueError):
        TfidfRanker().recommend(profile, [], top_k=1)


def test_recommend_includes_structured_reasons():
    profile = _profile()
    jobs = [_job("JOB-1")]
    results = TfidfRanker().recommend(profile, jobs, top_k=1)
    reasons = results[0].reasons
    assert any("Matched skills" in r for r in reasons)
