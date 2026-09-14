"""Sentence-embedding semantic ranker.

Loads `sentence-transformers/all-MiniLM-L6-v2` lazily and caches one instance
per process, so `/health` and TF-IDF-only usage never pay its load cost. The
heavy `sentence_transformers` import itself is deferred into the loader
function for the same reason.
"""

from __future__ import annotations

from typing import Any

from swetrack.domains.opportunities.models import CandidateProfile, JobRecord
from swetrack.domains.opportunities.preprocessing import build_candidate_text, build_job_text
from swetrack.domains.opportunities.ranking.base import Ranker, ScoredJob

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Lazily load and cache one SentenceTransformer instance per process."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class EmbeddingRanker(Ranker):
    """Semantic ranker: sentence-transformer embeddings scored by cosine similarity."""

    name = "embedding"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name

    def score_jobs(self, profile: CandidateProfile, jobs: list[JobRecord]) -> list[ScoredJob]:
        """Encode candidate/job text into normalized embeddings and score by dot product."""
        model = _get_model(self.model_name)
        candidate_text = build_candidate_text(profile)
        job_texts = [build_job_text(job) for job in jobs]

        embeddings = model.encode(
            [candidate_text, *job_texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        candidate_vector = embeddings[0]
        job_vectors = embeddings[1:]
        # Vectors are normalized, so dot product equals cosine similarity.
        similarities = job_vectors @ candidate_vector

        return [ScoredJob(job=job, score=float(score)) for job, score in zip(jobs, similarities)]
