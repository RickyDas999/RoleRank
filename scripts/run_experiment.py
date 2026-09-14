"""Run TF-IDF and embedding rankers on the example dataset and log to MLflow.

Usage:
    python scripts/run_experiment.py --k 5

Uses a local file-backed MLflow tracking store under `mlruns/` (gitignored).
For each ranker, logs its configuration, K, job count, git commit,
Precision@K, NDCG@K, and the ranked top-K results as a JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow

from swetrack.domains.opportunities.config import (
    DEFAULT_JOBS_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_PROFILE_PATH,
    load_candidate_profile,
    load_jobs,
    load_relevance_labels,
)
from swetrack.domains.opportunities.evaluation import ndcg_at_k, precision_at_k
from swetrack.domains.opportunities.models import CandidateProfile, JobRecord
from swetrack.domains.opportunities.ranking.base import Ranker
from swetrack.domains.opportunities.ranking.embeddings import EmbeddingRanker
from swetrack.domains.opportunities.ranking.tfidf import TfidfRanker

ROOT_DIR = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT_DIR / "mlruns"
EXPERIMENT_NAME = "swetrack-milestone-1"


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare TF-IDF and embedding rankers.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    return parser.parse_args(argv)


def run_ranker_experiment(
    ranker: Ranker,
    profile: CandidateProfile,
    jobs: list[JobRecord],
    labels: dict[str, int],
    k: int,
    params: dict[str, object],
) -> tuple[float, float, str]:
    """Rank all jobs once, compute Precision@K/NDCG@K, and log an MLflow run."""
    full_ranking = ranker.recommend(profile, jobs, top_k=len(jobs))
    ranked_ids = [item.job.job_id for item in full_ranking]

    precision = precision_at_k(ranked_ids, labels, k)
    ndcg = ndcg_at_k(ranked_ids, labels, k)
    top_k_results = full_ranking[:k]

    with mlflow.start_run(run_name=ranker.name) as run:
        mlflow.log_param("ranker", ranker.name)
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("k", k)
        mlflow.log_param("job_count", len(jobs))
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_metric("precision_at_k", precision)
        mlflow.log_metric("ndcg_at_k", ndcg)

        artifact = {
            "ranker": ranker.name,
            "k": k,
            "precision_at_k": precision,
            "ndcg_at_k": ndcg,
            "top_k": [
                {
                    "rank": item.rank,
                    "job_id": item.job.job_id,
                    "title": item.job.title,
                    "score": item.score,
                }
                for item in top_k_results
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / f"{ranker.name}_top_{k}.json"
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact_path))

        run_id = run.info.run_id

    return precision, ndcg, run_id


def main(argv: list[str] | None = None) -> None:
    """Load data, run both rankers, log to MLflow, and print a summary."""
    args = parse_args(argv)
    profile = load_candidate_profile(args.profile)
    jobs = load_jobs(args.jobs)
    labels = load_relevance_labels(args.labels)

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    tfidf_ranker = TfidfRanker()
    tfidf_precision, tfidf_ndcg, tfidf_run_id = run_ranker_experiment(
        tfidf_ranker,
        profile,
        jobs,
        labels,
        args.k,
        params={"ngram_range": str(tfidf_ranker.ngram_range), "stop_words": tfidf_ranker.stop_words},
    )
    print(f"[tfidf] Precision@{args.k}={tfidf_precision:.4f} NDCG@{args.k}={tfidf_ndcg:.4f} run_id={tfidf_run_id}")

    embedding_ranker = EmbeddingRanker()
    embed_precision, embed_ndcg, embed_run_id = run_ranker_experiment(
        embedding_ranker,
        profile,
        jobs,
        labels,
        args.k,
        params={"model_name": embedding_ranker.model_name},
    )
    print(f"[embedding] Precision@{args.k}={embed_precision:.4f} NDCG@{args.k}={embed_ndcg:.4f} run_id={embed_run_id}")


if __name__ == "__main__":
    main()
