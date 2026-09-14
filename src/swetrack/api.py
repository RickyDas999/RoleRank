"""FastAPI service: GET /health, GET /jobs, and POST /recommend.

The sentence-embedding model is never touched by /health or /jobs. It is
only loaded, lazily and cached once per process, the first time a
POST /recommend request selects ranker="embedding" (see
swetrack.ranking.embeddings).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from swetrack import __version__
from swetrack.config import DataLoadError, load_candidate_profile, load_jobs
from swetrack.models import HealthResponse, JobRecord, RecommendRequest, RecommendResponse
from swetrack.ranking.base import Ranker
from swetrack.ranking.embeddings import EmbeddingRanker
from swetrack.ranking.tfidf import TfidfRanker

app = FastAPI(title="SWETrack API", version=__version__)


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
