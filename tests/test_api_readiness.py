"""Tests for GET /opportunities/{job_id}/readiness.

Overrides the API's DB session dependency with an isolated SQLite DB so
these tests never touch the real persistent var/swetrack.db file that
swetrack.api opens at import time for local/dev use.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from swetrack.api import app, get_db_session
from swetrack.domains.opportunities.config import DEFAULT_JOBS_PATH, load_jobs
from swetrack.infrastructure.database.base import get_engine, get_sessionmaker, init_db


@pytest.fixture
def client(tmp_path):
    """A TestClient wired to an isolated, file-based SQLite DB (not ``:memory:``).

    TestClient runs the sync endpoint in a worker thread. SQLAlchemy's
    ``SingletonThreadPool`` for ``sqlite:///:memory:`` hands each thread its
    own separate, schema-less in-memory database, so the shared
    ``db_session`` fixture (fine for direct, single-threaded service-layer
    tests) fails here with "no such table". A temp file sidesteps that:
    every thread opens its own connection to the same on-disk file.
    """
    engine = get_engine(f"sqlite:///{tmp_path / 'test_readiness.db'}")
    init_db(engine)
    session = get_sessionmaker(engine)()

    def _override_get_db_session():
        yield session

    app.dependency_overrides[get_db_session] = _override_get_db_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        session.close()
        engine.dispose()


def test_readiness_endpoint_returns_fit_and_readiness_scores(client):
    known_job_id = load_jobs(DEFAULT_JOBS_PATH)[0].job_id

    response = client.get(f"/opportunities/{known_job_id}/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == known_job_id
    assert isinstance(body["fit_score"], float)
    assert 0.0 <= body["readiness_score"] <= 1.0
    assert isinstance(body["skill_gaps"], list)
    assert isinstance(body["recommended_activities"], list)


def test_readiness_endpoint_404s_for_unknown_job(client):
    response = client.get("/opportunities/DOES-NOT-EXIST/readiness")
    assert response.status_code == 404


def test_readiness_endpoint_rejects_unknown_ranker(client):
    known_job_id = load_jobs(DEFAULT_JOBS_PATH)[0].job_id

    response = client.get(f"/opportunities/{known_job_id}/readiness", params={"ranker": "not-a-real-ranker"})

    assert response.status_code == 422
