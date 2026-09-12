"""CLI: rank sample jobs against a candidate profile and print the top K.

Usage:
    python scripts/recommend.py --ranker tfidf --top-k 5
    python scripts/recommend.py --ranker tfidf --top-k 5 --profile path/to/profile.yaml

The example candidate profile (config/candidate_profile.example.yaml) is used
by default. It is never assumed to be a real resume; callers pass --profile
explicitly to use their own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rolerank.config import DEFAULT_JOBS_PATH, DEFAULT_PROFILE_PATH, load_candidate_profile, load_jobs
from rolerank.models import RecommendationItem
from rolerank.ranking.base import Ranker
from rolerank.ranking.embeddings import EmbeddingRanker
from rolerank.ranking.tfidf import TfidfRanker

_RANKER_CHOICES = ("tfidf", "embedding")


def build_ranker(name: str) -> Ranker:
    """Construct a ranker by name."""
    if name == "tfidf":
        return TfidfRanker()
    if name == "embedding":
        return EmbeddingRanker()
    raise ValueError(f"Unknown ranker: {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Rank sample jobs against a candidate profile.")
    parser.add_argument("--ranker", choices=_RANKER_CHOICES, default="tfidf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    return parser.parse_args(argv)


def format_item(item: RecommendationItem) -> str:
    """Render one recommendation item as human-readable text."""
    lines = [
        f"{item.rank}. [{item.job.job_id}] {item.job.title} @ {item.job.company} "
        f"(ranker={item.ranker}, score={item.score:.4f})",
        f"   Location: {item.job.location or 'n/a'} | Experience: {item.job.experience_level or 'n/a'}",
    ]
    lines.extend(f"   - {reason}" for reason in item.reasons)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Load data, rank jobs, and print the top-K recommendations."""
    args = parse_args(argv)
    profile = load_candidate_profile(args.profile)
    jobs = load_jobs(args.jobs)
    ranker = build_ranker(args.ranker)
    results = ranker.recommend(profile, jobs, args.top_k)

    for item in results:
        print(format_item(item))
        print()


if __name__ == "__main__":
    main()
