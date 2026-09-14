"""TF-IDF lexical baseline ranker.

Fits a fresh `TfidfVectorizer` per request over the candidate text plus the
current job corpus. This is acceptable for a small demonstration corpus (see
docs/ml-design.md); at production scale, fit a versioned vectorizer on the
job corpus offline and transform each request against it at inference time.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from swetrack.models import CandidateProfile, JobRecord
from swetrack.preprocessing import build_candidate_text, build_job_text
from swetrack.ranking.base import Ranker, ScoredJob


class TfidfRanker(Ranker):
    """Lexical baseline: TF-IDF vectors over unigrams/bigrams with cosine similarity."""

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), stop_words: str = "english") -> None:
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def score_jobs(self, profile: CandidateProfile, jobs: list[JobRecord]) -> list[ScoredJob]:
        """Fit TF-IDF on [candidate_text, *job_texts] and score by cosine similarity."""
        candidate_text = build_candidate_text(profile)
        job_texts = [build_job_text(job) for job in jobs]

        vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
        matrix = vectorizer.fit_transform([candidate_text, *job_texts])

        candidate_vector = matrix[0:1]
        job_vectors = matrix[1:]
        similarities = cosine_similarity(candidate_vector, job_vectors)[0]

        return [ScoredJob(job=job, score=float(score)) for job, score in zip(jobs, similarities)]
