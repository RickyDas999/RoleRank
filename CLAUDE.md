# CLAUDE.md — RoleRank Engineering Contract

## Project context

RoleRank is a personalized, content-based recommendation system that ranks new-grad software-engineering jobs against a candidate's skills, experience, and preferences. Ricky is a new-grad software engineer with strong Python/Java/AWS/backend experience who is adding hands-on ML engineering experience to his portfolio.

This first milestone is intentionally optimized for speed, credibility, interview defensibility, and $0 cost. It must produce working software, not speculative architecture.

## Required reading order

Before changing code, read these files completely:

1. `docs/product-requirements.md`
2. `docs/ml-design.md`
3. `docs/milestone-1.md`
4. `docs/interview-notes.md`

Treat them as the source of truth. If they conflict, use this precedence: `CLAUDE.md` → `docs/milestone-1.md` → `docs/ml-design.md` → `docs/product-requirements.md`.

## Non-negotiable constraints

- Spend $0. Use only local, open-source software and models.
- Do not call paid APIs, require API keys, or provision AWS/cloud resources.
- Do not scrape job sites in Milestone 1. Use a small checked-in demonstration dataset with source metadata clearly marked as sample or synthetic.
- Do not fabricate metrics. Compute and record every reported number from checked-in data and code.
- Do not claim a feature is implemented merely because an interface or placeholder exists.
- Never commit a real resume, private contact information, credentials, `.env`, MLflow database, downloaded model weights, caches, or large generated artifacts.
- Keep the system modular so TF-IDF and embedding rankers use one common interface.
- Favor simple, readable Python over premature abstractions.
- Add type hints and concise docstrings to public functions and classes.
- Pin or bound dependencies to compatible versions. Use Python 3.11.
- Use deterministic seeds wherever applicable.

## Required architecture

Use a `src` layout with a package named `rolerank`. The implementation should contain these responsibilities, although exact filenames may change when justified:

```text
rolerank/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── config/
│   └── candidate_profile.example.yaml
├── data/
│   ├── sample_jobs.csv
│   └── relevance_labels.csv
├── docs/
│   ├── product-requirements.md
│   ├── ml-design.md
│   ├── milestone-1.md
│   └── interview-notes.md
├── mlruns/                 # generated locally; gitignored
├── scripts/
│   ├── recommend.py
│   └── run_experiment.py
├── src/rolerank/
│   ├── api.py
│   ├── config.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── ranking/
│       ├── base.py
│       ├── tfidf.py
│       └── embeddings.py
└── tests/
    ├── test_api.py
    ├── test_evaluation.py
    ├── test_preprocessing.py
    └── test_rankers.py
```

## Implementation expectations

- Represent candidate and job inputs with validated Pydantic models.
- Normalize text consistently but do not over-clean it. Preserve technology tokens such as `C++`, `.NET`, `Node.js`, and `AWS` as much as practical.
- Combine semantically named job fields into ranking text; do not silently include protected or irrelevant attributes.
- Implement `TfidfRanker` with scikit-learn `TfidfVectorizer`, unigrams and bigrams, English stop-word removal, and cosine similarity.
- Implement `EmbeddingRanker` with the free `sentence-transformers/all-MiniLM-L6-v2` model. Load it lazily and cache one process-level instance. Use normalized embeddings so dot product or cosine similarity is well-defined.
- Return scores in descending order with deterministic tie-breaking by job ID.
- Return transparent explanations based on structured skill overlap and matched preferences. Do not describe embedding dimensions as human-interpretable feature importance.
- Implement evaluation for `Precision@K` and `NDCG@K` from checked-in relevance labels.
- Log ranker name, model/config parameters, metrics, and a small result artifact to a local MLflow file store.
- Expose `GET /health`, `GET /jobs`, and `POST /recommend` through FastAPI.
- Make the API usable without a private resume by defaulting to the example candidate profile only when explicitly requested by the caller or CLI.
- Ensure Docker health and API commands are documented.

## Development workflow

1. Inspect the repository and restate a short implementation plan.
2. Implement the smallest vertical slice first: load data → TF-IDF rank → CLI output.
3. Add embeddings, evaluation, experiment tracking, API, tests, and Docker in that order.
4. Run formatting/linting only if configured; never substitute static checks for runtime verification.
5. Run every acceptance command in `docs/milestone-1.md`.
6. Fix failures before reporting completion.
7. Replace documentation placeholders with actual measured results and commands.
8. **Never run `git commit` or `git push`. The user creates every commit themselves.**

### Checkpoint protocol

Work through Milestone 1 in the following incremental, reviewable checkpoints. Do not skip ahead to a later checkpoint before the current one is confirmed by the user.

1. Project setup, sample data, schemas, and preprocessing
2. TF-IDF ranking pipeline and CLI
3. Sentence-embedding ranker, evaluation metrics, and MLflow
4. FastAPI endpoints and API tests
5. Docker, full test suite, README, and final verification

After completing and testing each checkpoint:

1. Stop and summarize what was implemented.
2. List the changed files.
3. Show the relevant test/verification results.
4. Provide one concise conventional commit message.
5. Wait for the user to create the commit and confirm before continuing to the next checkpoint.

## Scope exclusions for Milestone 1

Do not add React, authentication, a production database, queues, Kubernetes, Terraform, AWS CDK, LLM APIs, collaborative filtering, feedback-based training, or scheduled retraining. Record worthwhile ideas under Future Work rather than implementing them.

## Communication requirements

If blocked, identify the exact failing command, error, likely cause, and safest next step. At completion, return the exact report defined in `docs/milestone-1.md`; do not merely say that everything works.
