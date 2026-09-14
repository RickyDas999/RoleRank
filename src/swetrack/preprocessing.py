"""Text normalization and candidate/job text construction.

Normalization is intentionally light: it collapses whitespace and drops
blank entries without stripping punctuation, so technology tokens such as
``C++``, ``.NET``, ``Node.js``, and ``AWS`` survive intact for both the
TF-IDF and embedding rankers.
"""

from __future__ import annotations

import re

from swetrack.models import CandidateProfile, JobRecord

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Collapse whitespace and strip leading/trailing space; never raises on None."""
    if text is None:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", str(text))
    return collapsed.strip()


def normalize_list(items: list[str] | None) -> list[str]:
    """Normalize a list of short strings, dropping blank entries."""
    if not items:
        return []
    normalized = (normalize_text(item) for item in items)
    return [item for item in normalized if item]


def build_candidate_text(profile: CandidateProfile) -> str:
    """Build the candidate query document from named, reproducible sections."""
    skills = ", ".join(normalize_list(profile.skills))
    experience = normalize_text(profile.experience)
    roles = ", ".join(normalize_list(profile.preferred_roles))
    interests = ", ".join(normalize_list(profile.keywords))
    return (
        f"skills: {skills}\n"
        f"experience: {experience}\n"
        f"preferred roles: {roles}\n"
        f"interests: {interests}"
    )


def build_job_text(job: JobRecord) -> str:
    """Build a job document from named, reproducible sections."""
    title = normalize_text(job.title)
    description = normalize_text(job.description)
    skills = ", ".join(normalize_list(job.skills))
    experience_level = normalize_text(job.experience_level)
    return (
        f"title: {title}\n"
        f"description: {description}\n"
        f"skills: {skills}\n"
        f"experience level: {experience_level}"
    )
