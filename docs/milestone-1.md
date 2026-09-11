# Milestone 1 — 48-Hour Resume-Ready Vertical Slice

## Objective

Deliver a working end-to-end job-ranking service with two comparable content-based rankers, offline evaluation, local experiment tracking, an API, tests, and Docker. Do not stop at scaffolding.

## Implementation checklist

- [ ] Create the Python 3.11 project with a `src` package layout.
- [ ] Add 20–30 clearly labeled sample/synthetic job records.
- [ ] Add the example candidate profile using only professional, non-sensitive information.
- [ ] Add manually curated ordinal relevance labels and document their subjectivity.
- [ ] Implement validated data loading and missing-value handling.
- [ ] Implement shared preprocessing and ranking result types.
- [ ] Implement deterministic TF-IDF ranking.
- [ ] Implement lazy-loaded local sentence-embedding ranking.
- [ ] Generate structured overlap explanations.
- [ ] Implement Precision@K and NDCG@K with tests.
- [ ] Run both rankers and log actual experiments locally with MLflow.
- [ ] Implement `GET /health`, `GET /jobs`, and `POST /recommend`.
- [ ] Add unit/integration tests covering preprocessing, ordering, top K, evaluation, validation errors, and API schema.
- [ ] Add Dockerfile, `.dockerignore`, and verified run instructions.
- [ ] Write a README with problem, architecture, quick start, API examples, experiment results, limitations, cost stance, and roadmap.
- [ ] Run every acceptance check below and fix failures.

## Acceptance checks

Claude may adapt environment activation syntax to the user's OS, but the underlying commands and evidence are required.

### 1. Environment and tests

```bash
python --version
python -m pip install -e ".[dev]"
python -m pytest -q
```

Required: Python 3.11.x and all tests pass. If the local machine uses another compatible Python version, document it rather than concealing it.

### 2. TF-IDF CLI

```bash
python scripts/recommend.py --ranker tfidf --top-k 5
```

Required: five jobs in descending score order, with IDs and explanations.

### 3. Embedding CLI

```bash
python scripts/recommend.py --ranker embedding --top-k 5
```

Required: five jobs in descending score order. The first run may download the free local model.

### 4. Reproducible experiment

```bash
python scripts/run_experiment.py --k 5
```

Required: actual Precision@5 and NDCG@5 for both rankers, MLflow run IDs, and saved top-five artifacts. Copy measured results into the README; do not predict which model wins.

### 5. API

```bash
uvicorn rolerank.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" --data @examples/recommend_request.json
```

Required: healthy JSON response and a valid top-K recommendation response. On Windows PowerShell, use `curl.exe` or `Invoke-RestMethod` and document the exact working command.

### 6. Docker

```bash
docker build -t rolerank:milestone-1 .
docker run --rm -p 8000:8000 rolerank:milestone-1
```

Required: `/health` and at least one TF-IDF recommendation work from the container. The image should not pre-download or bundle large model caches unless explicitly justified. Document how embedding mode obtains or mounts its model cache.

### 7. Repository hygiene

```bash
git status --short
git ls-files
```

Required: no secrets, private resume, `.env`, virtual environment, cache, downloaded weights, `mlruns/`, or large generated artifact is tracked.

## Minimum test cases

- empty or whitespace fields are normalized safely;
- missing required fields fail clearly;
- ranking returns exactly K unique jobs;
- scores are non-increasing;
- ties use stable job-ID ordering;
- a deliberately strong exact-match fixture outranks an unrelated fixture for TF-IDF;
- evaluation functions match hand-calculated small examples;
- invalid ranker and invalid K return 4xx API responses;
- `/health` does not load the sentence-transformer model;
- API response matches its declared schema.

Avoid brittle tests asserting the exact embedding order of many sample jobs across dependency/model changes. Test invariants and limited semantic fixtures instead.

## Required completion report from Claude Code

Return this exact structure with real output, not summaries such as “tests passed”:

```text
MILESTONE 1 COMPLETION REPORT

1. Status
- Complete / Blocked
- Remaining limitations:

2. Repository
- Commit hash:
- Final tree (depth 3):

3. Verification
- Python version:
- pytest command and full summary line:
- TF-IDF CLI top 5:
- Embedding CLI top 5:
- Precision@5 and NDCG@5 for each ranker:
- MLflow run IDs:
- API health response:
- Sample POST /recommend response:
- Docker build result:
- Docker API result:

4. Design deviations
- Any deviation from the handoff documents and why:

5. Files to review
- Core ranking files:
- Evaluation/experiment files:
- API files:
- README:

6. Git hygiene
- git status --short:
- Confirmation that no secrets/private resume/model weights/mlruns are tracked:
```

Also paste the final README and `pyproject.toml` if they are short enough; otherwise provide their relevant sections.

## Resume threshold

After the evidence is reviewed, the truthful Milestone 1 project entry may describe implemented TF-IDF and sentence-embedding ranking, local MLflow experiments, FastAPI serving, tests, and Docker. It may not say “deployed on AWS,” “learned from user behavior,” “production monitoring,” or quote an improvement until those exist and are measured.
