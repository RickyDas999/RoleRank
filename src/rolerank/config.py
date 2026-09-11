"""Loading of on-disk configuration: candidate profile YAML and job/label CSVs.

All default paths are resolved relative to the repository root so the CLI
scripts and API work the same regardless of the caller's current directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from rolerank.models import CandidateProfile, JobRecord

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = ROOT_DIR / "config" / "candidate_profile.example.yaml"
DEFAULT_JOBS_PATH = ROOT_DIR / "data" / "sample_jobs.csv"
DEFAULT_LABELS_PATH = ROOT_DIR / "data" / "relevance_labels.csv"


class DataLoadError(ValueError):
    """Raised when on-disk data fails validation, with a clear cause."""


def load_candidate_profile(path: str | Path = DEFAULT_PROFILE_PATH) -> CandidateProfile:
    """Load and validate a candidate profile from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Candidate profile file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return CandidateProfile.model_validate(raw)
    except ValidationError as exc:
        raise DataLoadError(f"Invalid candidate profile at {path}: {exc}") from exc


def load_jobs(path: str | Path = DEFAULT_JOBS_PATH) -> list[JobRecord]:
    """Load and validate job records from a CSV file, failing clearly on bad rows."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Jobs file not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    jobs: list[JobRecord] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            job = JobRecord.model_validate(row)
        except ValidationError as exc:
            raise DataLoadError(f"Invalid job row at {path}:{row_number}: {exc}") from exc
        if job.job_id in seen_ids:
            raise DataLoadError(f"Duplicate job_id '{job.job_id}' at {path}:{row_number}")
        seen_ids.add(job.job_id)
        jobs.append(job)

    if not jobs:
        raise DataLoadError(f"No job records found in {path}")
    return jobs


def load_relevance_labels(path: str | Path = DEFAULT_LABELS_PATH) -> dict[str, int]:
    """Load ordinal relevance labels (0/1/2) keyed by job_id."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Relevance labels file not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    required_cols = {"job_id", "relevance"}
    if not required_cols.issubset(frame.columns):
        raise DataLoadError(f"Relevance labels file {path} must have columns {sorted(required_cols)}")

    labels: dict[str, int] = {}
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        job_id = str(row["job_id"]).strip()
        if not job_id:
            raise DataLoadError(f"Blank job_id in relevance labels at {path}:{row_number}")
        try:
            relevance = int(row["relevance"])
        except (TypeError, ValueError) as exc:
            raise DataLoadError(
                f"Non-integer relevance value at {path}:{row_number}: {row['relevance']!r}"
            ) from exc
        if relevance not in (0, 1, 2):
            raise DataLoadError(f"Relevance value out of range [0,2] at {path}:{row_number}: {relevance}")
        labels[job_id] = relevance
    return labels
