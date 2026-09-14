# SWETrack ML Design — Milestone 1 (Opportunity Intelligence)

## Framing

Milestone 1 is a content-based information-retrieval and ranking system. It is not yet collaborative filtering, supervised preference learning, or learning-to-rank.

For a candidate query/profile \(q\) and each job document \(d_i\), a ranker produces a similarity score:

\[
s_i = \cos(q, d_i) = \frac{q \cdot d_i}{\lVert q \rVert\lVert d_i\rVert}
\]

Jobs are sorted by descending \(s_i\). Similarity scores are not probabilities.

## Text construction

Build candidate text from named sections so the representation is reproducible:

```text
skills: ...
experience: ...
preferred roles: ...
interests: ...
```

Build job text from:

```text
title: ...
description: ...
skills: ...
experience level: ...
```

Location preference should be used as an explanation and, only if clearly documented, a small structured score adjustment. For the first comparison, prefer pure text similarity so both baselines are easy to interpret. Never include company prestige, gendered signals, age, ethnicity, or other protected attributes.

## Ranker A: TF-IDF baseline

Use `TfidfVectorizer(stop_words="english", ngram_range=(1, 2))` over the candidate and current job corpus.

- Term frequency represents occurrence within a document.
- Inverse document frequency reduces the weight of terms common across the corpus.
- Unigrams capture individual terms; bigrams capture phrases such as `machine learning` or `distributed systems`.
- The resulting matrix is sparse because each document uses a small fraction of the corpus vocabulary.

For a demonstration-sized corpus, fitting per request is acceptable but document the production alternative: fit a versioned vectorizer on the job corpus and transform requests at inference time.

## Ranker B: sentence embeddings

Use `sentence-transformers/all-MiniLM-L6-v2` locally. It is a compact general-purpose sentence embedding model suitable for semantic similarity demonstrations.

- Encode candidate and job text into dense vectors.
- Normalize embeddings.
- Compute cosine similarity or an equivalent dot product on normalized vectors.
- Load the model lazily and reuse it across requests.
- Document the model name as a configurable parameter.

The first local run downloads free model weights. No hosted inference API is permitted. Model weights and caches must not be committed.

## Explanations

Explanations must use transparent structured comparisons, for example:

- `Matched skills: Python, AWS, FastAPI`
- `Preferred role match: Backend`
- `Preferred location match: New York`

Do not infer explanations by claiming particular embedding coordinates correspond to skills. Semantic similarity and structured explanations are separate outputs.

## Evaluation data

Create 20–30 varied sample job records spanning:

- relevant entry-level backend/general SWE/full-stack roles;
- partially relevant data/ML/cloud roles;
- clearly irrelevant senior, mobile-only, design, or non-engineering roles.

Create `relevance_labels.csv` with explicit ordinal judgments for the example profile:

- `0`: irrelevant;
- `1`: somewhat relevant;
- `2`: strongly relevant.

These labels are manually curated demonstration judgments, not objective ground truth. Document this limitation.

## Metrics

Compute both metrics at a documented K, defaulting to K=5:

### Precision@K

Treat relevance label greater than zero as relevant:

\[
P@K = \frac{\text{relevant jobs in top K}}{K}
\]

### NDCG@K

Use graded relevance and standard discounted cumulative gain:

\[
DCG@K = \sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}
\]

\[
NDCG@K = \frac{DCG@K}{IDCG@K}
\]

Handle the zero-ideal-DCG case safely.

The dataset is tiny and tailored to one example profile, so metrics demonstrate evaluation plumbing rather than statistical generalization. The README must say this.

## MLflow experiment

The experiment script must run both rankers on the same version of the example profile, jobs, and labels. For each run, log:

- ranker name;
- model/vectorizer configuration;
- K;
- job count;
- Precision@K;
- NDCG@K;
- ranked top-K results as an artifact;
- dataset fingerprints or Git commit if readily available.

Use a local file-backed tracking URI. The generated `mlruns/` directory is gitignored.

## Honest interpretation

An embedding model outperforming TF-IDF on the tiny demonstration set is not guaranteed. Report the actual result. A tie or weaker result is still useful: explain potential causes such as labels, sample size, domain mismatch, truncation, or strong exact skill vocabulary.

## Later milestones, not current claims

- SQLite interaction store with apply/interested/skip feedback;
- logistic-regression preference model;
- hybrid ranker combining semantic and structured features;
- learning-to-rank after enough interactions;
- larger legally sourced dataset and held-out evaluation;
- drift and service monitoring;
- cloud deployment only if a genuinely free and safe option is deliberately selected.
