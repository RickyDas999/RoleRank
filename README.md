# SWETrack

SWETrack is a personalized ML platform for new-grad SWE recruiting and interview
preparation. This repository currently implements its **Opportunity Intelligence**
subsystem (formerly RoleRank): a content-based recommendation service that ranks
new-grad software-engineering jobs against a candidate's skills, experience, and
preferences. This is **Milestone 1**: a $0, fully local, resume-ready vertical slice —
not a deployed product and not a system that has learned from real user behavior yet.

## Problem

New-grad applicants review many postings with inconsistent titles and long descriptions.
Keyword search can miss semantically relevant jobs, while generic job boards do not
understand an individual candidate's technical background or role preferences. SWETrack's
Opportunity Intelligence subsystem ranks a supplied set of software-engineering jobs
against a supplied candidate profile and explains the most visible matching signals.

Opportunity Intelligence is currently an **unsupervised content-based retrieval/ranking
system**. It is not collaborative filtering, not a supervised model trained on user
feedback, and its similarity scores are not probabilities of getting an interview.

## Architecture

```text
rolerank/
├── config/
│   └── candidate_profile.example.yaml   # example, non-sensitive profile
├── data/
│   ├── sample_jobs.csv                  # 26 sample/synthetic job postings
│   └── relevance_labels.csv             # manually curated 0/1/2 relevance labels
├── examples/
│   └── recommend_request.json           # sample POST /recommend body
├── src/rolerank/
│   ├── api.py                           # FastAPI: /health, /jobs, /recommend
│   ├── config.py                        # YAML/CSV loading + validation
│   ├── models.py                        # Pydantic schemas (candidate, job, API I/O)
│   ├── preprocessing.py                 # text normalization + document construction
│   ├── evaluation.py                    # Precision@K, NDCG@K
│   └── ranking/
│       ├── base.py                      # shared Ranker interface, sorting, explanations
│       ├── tfidf.py                     # TfidfRanker (lexical baseline)
│       └── embeddings.py                # EmbeddingRanker (sentence-transformers)
├── scripts/
│   ├── recommend.py                     # CLI: rank sample jobs, print top K
│   └── run_experiment.py                # run both rankers, log to MLflow
├── tests/                               # pytest: preprocessing, rankers, evaluation, API
├── Dockerfile / .dockerignore
└── mlruns/                              # generated locally by run_experiment.py; gitignored
```

Both rankers implement one `Ranker` interface (`score_jobs` → sort, deterministic
job-ID tie-break, top-K truncation, and structured explanations are handled once in
`ranking/base.py`), so they stay directly comparable.

**Candidate text** is built from named sections: `skills`, `experience`,
`preferred roles`, `interests`. **Job text** is built from `title`, `description`,
`skills`, `experience level`. Company, location, and other attributes are never folded
into the ranked text; location is only used for structured match explanations.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Rank the sample jobs against the example candidate profile:

```bash
python scripts/recommend.py --ranker tfidf --top-k 5
python scripts/recommend.py --ranker embedding --top-k 5   # first run downloads the free local model
```

Run the offline experiment (both rankers, logged to a local MLflow file store):

```bash
python scripts/run_experiment.py --k 5
```

Serve the API:

```bash
uvicorn rolerank.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" --data @examples/recommend_request.json
```

On Windows PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" --data "@examples/recommend_request.json"
```

## API

- `GET /health` — service status and version. Never loads the sentence-embedding model.
- `GET /jobs` — sample job metadata, with optional `?limit=&offset=` pagination.
- `POST /recommend` — ranks jobs against a candidate profile.

`POST /recommend` never silently substitutes a private resume. Either supply `profile`
directly, or explicitly opt into the checked-in example profile with
`"use_example_profile": true`:

```json
{
  "use_example_profile": true,
  "ranker": "tfidf",
  "top_k": 5
}
```

Sample response (truncated to one result):

```json
{
  "ranker": "tfidf",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "job": {
        "job_id": "JOB-003",
        "company": "CloudForge Labs",
        "title": "Full Stack Software Engineer - New Grad",
        "location": "Remote",
        "skills": ["Python", "FastAPI", "React", "AWS", "Docker", "CI/CD", "Git"],
        "experience_level": "New graduate",
        "source": "synthetic"
      },
      "ranker": "tfidf",
      "score": 0.25091064589057305,
      "reasons": [
        "Matched skills: Python, AWS, Git, Docker, FastAPI",
        "Preferred role match: Software Engineer, Full Stack Engineer",
        "Preferred location match: Remote"
      ]
    }
  ]
}
```

Invalid `ranker`, non-positive `top_k`, `top_k` greater than the available job count, a
malformed profile, or an omitted profile with `use_example_profile` unset all return a
`422` with a clear error message.

Similarity scores are relative signals for ranking, not calibrated probabilities and not
a claim that a candidate will receive an interview.

## Experiment results

Measured by running `python scripts/run_experiment.py --k 5` against the checked-in
26-job sample dataset and its manually curated relevance labels:

| Ranker    | Precision@5 | NDCG@5 | MLflow run ID                      |
|-----------|-------------|--------|-------------------------------------|
| TF-IDF    | 1.0000      | 1.0000 | `00686e24e08349c28c6b8a0c5b1dcdda`  |
| Embedding | 1.0000      | 1.0000 | `06cf27e13c834de39167efcbef7a5813`  |

**Honest interpretation:** both rankers tie at Precision@5 = 1.0 and NDCG@5 = 1.0. The
top 5 jobs returned by both rankers all carry the strongest relevance label (2) in
`data/relevance_labels.csv`. This is not a fabricated or cherry-picked result — the
sample dataset has 10 clearly-worded, strongly-relevant entry-level backend/full-stack
postings that share exact technical vocabulary (Python, AWS, REST APIs, Docker) with the
example profile, so a lexical baseline is already strong here and there is no room for
either ranker to separate at K=5. An embedding model outperforming TF-IDF is not
guaranteed, and this run shows a tie rather than a win for either approach. The two
rankers do produce different orderings within the top 5 (compare the CLI outputs above),
reflecting TF-IDF's exact-term weighting versus the embedding model's semantic
similarity — the tie is in the *evaluation metric*, not in the raw output.

This is evaluation plumbing validated on a tiny, single-profile demonstration set, not
evidence that either ranker generalizes to a larger or different candidate population.

## Docker

```bash
docker build -t rolerank:milestone-1 .
docker run --rm -p 8000:8000 rolerank:milestone-1
```

Then, from the host:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" --data @examples/recommend_request.json
```

The image installs `sentence-transformers`/`torch` (a declared runtime dependency for the
embedding ranker) but does **not** pre-download the `all-MiniLM-L6-v2` model weights at
build time — `GET /health` and TF-IDF requests never touch the model. The first
`ranker="embedding"` request inside a running container downloads the weights from
Hugging Face into `/root/.cache/huggingface`, which is lost when the container is
removed. To persist/reuse that cache across runs, mount a volume:

```bash
docker run --rm -p 8000:8000 -v rolerank-hf-cache:/root/.cache/huggingface rolerank:milestone-1
```

> Docker build/run were not executed in this development environment (Docker Desktop was
> not installed at the time of this milestone) — see the completion report for the exact
> blocked commands. The Dockerfile/`.dockerignore` follow the same install and run path
> used to verify the API locally above, and should be re-verified with `docker build` /
> `docker run` before relying on this section as confirmed.

## Cost stance

Everything in this milestone runs locally and costs $0: TF-IDF is pure scikit-learn;
the embedding model is a free, locally-run Hugging Face checkpoint; experiment tracking
uses a local MLflow file store; there are no paid APIs, API keys, or provisioned cloud
resources.

## Limitations

- Small (26-job), hand-curated, synthetic/sample demonstration dataset — not scraped or
  production data.
- Relevance labels are subjective ordinal judgments from one example candidate profile,
  not objective ground truth.
- No held-out user population, no online A/B test.
- The embedding model is general-purpose, not fine-tuned for job matching.
- No learned personalization or interaction feedback yet (see Future Work).
- Local, file-backed experiment store and a single-process API — not a production
  deployment or model registry.
- TF-IDF is fit fresh per request over the current corpus, which is fine at this scale;
  production use would fit and version a vectorizer offline and transform requests
  against it.

## Future work

Not implemented in Milestone 1 — recorded here rather than half-built:

- SQLite interaction store capturing explicit `apply` / `interested` / `skip` feedback.
- A learned, interpretable preference model (e.g. logistic regression over semantic +
  structured features) compared against both content baselines.
- A hybrid ranker combining semantic similarity with structured features.
- Learning-to-rank once enough interaction data exists.
- A larger, legally sourced dataset with held-out evaluation.
- Drift and service monitoring.
- Cloud deployment, only if a genuinely free and safe option is deliberately selected.
