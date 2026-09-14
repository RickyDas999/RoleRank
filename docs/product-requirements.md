# SWETrack Product Requirements — Milestone 1 (Opportunity Intelligence)

## Problem

New-grad applicants review many postings with inconsistent titles and long descriptions. Keyword search can miss semantically relevant jobs, while generic job boards do not understand an individual candidate's technical background or role preferences.

SWETrack's Opportunity Intelligence subsystem ranks a supplied set of software-engineering jobs against a supplied candidate profile and explains the most visible matching signals.

## Primary user

The first user is Ricky, a new-grad software engineer targeting backend, general SWE, and full-stack roles. The system must remain generic: user-specific information belongs in an example/config input, not hard-coded ranking logic.

## User stories

- As a candidate, I can submit my skills, experience text, and preferences and receive the top K matching jobs.
- As a candidate, I can choose a lexical TF-IDF baseline or a semantic embedding ranker.
- As a candidate, I can see job metadata, a similarity score, and understandable match reasons.
- As an engineer, I can compare rankers using reproducible offline metrics.
- As an engineer, I can inspect locally logged experiment parameters, metrics, and artifacts.

## Inputs

Candidate profile:

- skills: list of technologies or competencies;
- experience: free text describing relevant work and projects;
- preferred roles: e.g. backend, software engineer, full stack;
- preferred locations: locations or remote;
- optional keywords/interests.

Job record:

- stable `job_id`;
- company;
- title;
- location;
- description;
- skills;
- experience level;
- URL or source reference;
- provenance marker such as `sample` or `synthetic`.

## Outputs

Each recommendation returns:

- rank;
- job metadata;
- ranker name;
- numeric similarity score;
- deterministic, human-readable reasons such as matching skills, preferred role, or preferred location.

Scores are relative similarity signals, not calibrated probabilities or claims that a candidate will receive an interview.

## API behavior

### `GET /health`

Returns service health and version without loading the embedding model.

### `GET /jobs`

Returns available job metadata with optional pagination if it remains simple.

### `POST /recommend`

Accepts candidate profile, `ranker` (`tfidf` or `embedding`), and `top_k`. Validates `1 <= top_k <= number of available jobs` and returns sorted recommendations.

Invalid rankers and malformed inputs must produce clear 4xx responses.

## Non-functional requirements

- Fully local and zero-cost.
- Reproducible setup on Python 3.11.
- Deterministic output for fixed inputs and versions.
- Unit and integration tests for core behavior.
- No secret or private candidate data in Git.
- Embedding model is lazy-loaded so health checks and TF-IDF usage remain fast.
- README distinguishes implemented behavior from planned behavior.

## Milestone success

Milestone 1 succeeds when both rankers produce sensible ordered results, offline metrics are computed from explicit relevance judgments, MLflow records the comparison locally, the API passes tests, and the Docker container serves a verified request.
