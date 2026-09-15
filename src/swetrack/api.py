"""FastAPI service: GET /health, GET /jobs, POST /recommend, and GET /opportunities/{id}/readiness.

The sentence-embedding model is never touched by /health or /jobs. It is
only loaded, lazily and cached once per process, the first time a
POST /recommend or GET /opportunities/{id}/readiness request selects
ranker="embedding" (see swetrack.domains.opportunities.ranking.embeddings).
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from swetrack import __version__
from swetrack.domains.opportunities.config import DataLoadError, load_candidate_profile, load_jobs
from swetrack.domains.opportunities.models import (
    HealthResponse,
    JobRecord,
    RankerName,
    RecommendRequest,
    RecommendResponse,
)
from swetrack.domains.opportunities.ranking.base import Ranker
from swetrack.domains.opportunities.ranking.embeddings import EmbeddingRanker
from swetrack.domains.opportunities.ranking.tfidf import TfidfRanker
from swetrack.domains.opportunities.readiness import ReadinessResult, compute_readiness
from swetrack.infrastructure.database.base import get_engine, get_sessionmaker, init_db

app = FastAPI(title="SWETrack API", version=__version__)

# Module-level so every request reuses one connection pool rather than
# opening a fresh engine per call; defaults to the local SQLite file under
# var/ (see infrastructure/database/base.py), overridable via
# SWETRACK_DATABASE_URL for e.g. Postgres later.
_engine = get_engine()
init_db(_engine)
_SessionLocal = get_sessionmaker(_engine)


def get_db_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _build_ranker(name: str) -> Ranker:
    """Construct a ranker by name. `name` is already Literal-validated by RecommendRequest."""
    if name == "embedding":
        return EmbeddingRanker()
    return TfidfRanker()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check. Does not load the sentence-transformer model."""
    return HealthResponse(version=__version__)


@app.get("/jobs", response_model=list[JobRecord])
def get_jobs(
    limit: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[JobRecord]:
    """Return available job metadata, optionally paginated with limit/offset."""
    try:
        jobs = load_jobs()
    except DataLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if limit is None:
        return jobs[offset:]
    return jobs[offset : offset + limit]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Rank jobs against a supplied (or explicitly requested example) candidate profile."""
    try:
        jobs = load_jobs()
    except DataLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if request.top_k > len(jobs):
        raise HTTPException(
            status_code=422,
            detail=f"top_k={request.top_k} exceeds available jobs ({len(jobs)}).",
        )

    profile = request.profile
    if profile is None:
        try:
            profile = load_candidate_profile()
        except DataLoadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    ranker = _build_ranker(request.ranker)
    results = ranker.recommend(profile, jobs, request.top_k)
    return RecommendResponse(ranker=request.ranker, top_k=request.top_k, results=results)


@app.get("/opportunities/{job_id}/readiness", response_model=ReadinessResult)
def get_readiness(
    job_id: str,
    ranker: RankerName = Query(default="tfidf"),
    session: Session = Depends(get_db_session),
) -> ReadinessResult:
    """Role Fit and Readiness for one job, against the example candidate profile and tracked mastery.

    There is no per-request candidate profile here (unlike POST /recommend):
    a GET request has no body, and this is a single-user local app with no
    auth/user concept (CLAUDE.md explicitly avoids that), so the example
    profile is the only candidate representation available.
    """
    try:
        jobs = load_jobs()
    except DataLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id!r}")

    try:
        profile = load_candidate_profile()
    except DataLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return compute_readiness(session, job=job, profile=profile, ranker=_build_ranker(ranker))
