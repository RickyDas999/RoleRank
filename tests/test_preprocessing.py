"""Tests for schemas, text normalization, and on-disk data loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swetrack.domains.opportunities.config import (
    DEFAULT_JOBS_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_PROFILE_PATH,
    DataLoadError,
    load_candidate_profile,
    load_jobs,
    load_relevance_labels,
)
from swetrack.domains.opportunities.models import CandidateProfile, JobRecord
from swetrack.domains.opportunities.preprocessing import build_candidate_text, build_job_text, normalize_list, normalize_text


def test_normalize_text_collapses_whitespace_and_strips():
    assert normalize_text("  Python   and\tAWS \n") == "Python and AWS"


def test_normalize_text_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalize_text_preserves_technology_tokens():
    text = "Experience with C++, .NET, Node.js, and AWS."
    normalized = normalize_text(text)
    for token in ("C++", ".NET", "Node.js", "AWS"):
        assert token in normalized


def test_normalize_list_drops_blank_entries():
    assert normalize_list(["Python", "  ", "", "AWS "]) == ["Python", "AWS"]
    assert normalize_list(None) == []


def test_candidate_profile_rejects_missing_skills():
    with pytest.raises(ValidationError):
        CandidateProfile(skills=[], experience="Built backend services.")


def test_candidate_profile_rejects_blank_experience():
    with pytest.raises(ValidationError):
        CandidateProfile(skills=["Python"], experience="   ")


def test_candidate_profile_normalizes_comma_separated_skills():
    profile = CandidateProfile(skills="Python, AWS,  , Java", experience="Backend engineer.")
    assert profile.skills == ["Python", "AWS", "Java"]


def test_job_record_rejects_blank_required_fields():
    with pytest.raises(ValidationError):
        JobRecord(job_id="", company="Acme", title="Engineer")


def test_job_record_defaults_optional_fields_safely():
    job = JobRecord(job_id="JOB-1", company="Acme", title="Engineer")
    assert job.location == ""
    assert job.skills == []
    assert job.source == "synthetic"


def test_build_candidate_text_has_named_sections():
    profile = CandidateProfile(
        skills=["Python", "AWS"],
        experience="Built REST APIs.",
        preferred_roles=["Backend Engineer"],
        keywords=["distributed systems"],
    )
    text = build_candidate_text(profile)
    assert text.startswith("skills: Python, AWS\n")
    assert "experience: Built REST APIs." in text
    assert "preferred roles: Backend Engineer" in text
    assert "interests: distributed systems" in text


def test_build_job_text_has_named_sections():
    job = JobRecord(
        job_id="JOB-1",
        company="Acme",
        title="Backend Engineer",
        description="Build APIs.",
        skills=["Python", "AWS"],
        experience_level="Entry-level",
    )
    text = build_job_text(job)
    assert text.startswith("title: Backend Engineer\n")
    assert "description: Build APIs." in text
    assert "skills: Python, AWS" in text
    assert "experience level: Entry-level" in text


def test_load_candidate_profile_example():
    profile = load_candidate_profile(DEFAULT_PROFILE_PATH)
    assert len(profile.skills) > 0
    assert profile.experience


def test_load_jobs_sample_dataset():
    jobs = load_jobs(DEFAULT_JOBS_PATH)
    assert 20 <= len(jobs) <= 30
    job_ids = {job.job_id for job in jobs}
    assert len(job_ids) == len(jobs)
    assert all(job.source in ("sample", "synthetic") for job in jobs)


def test_load_relevance_labels_covers_all_jobs():
    jobs = load_jobs(DEFAULT_JOBS_PATH)
    labels = load_relevance_labels(DEFAULT_LABELS_PATH)
    job_ids = {job.job_id for job in jobs}
    assert set(labels.keys()) == job_ids
    assert all(v in (0, 1, 2) for v in labels.values())


def test_load_jobs_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(DataLoadError):
        load_jobs(tmp_path / "does_not_exist.csv")


def test_load_jobs_missing_required_column_raises(tmp_path):
    bad_csv = tmp_path / "bad_jobs.csv"
    bad_csv.write_text("company,title\nAcme,Engineer\n", encoding="utf-8")
    with pytest.raises(DataLoadError):
        load_jobs(bad_csv)
