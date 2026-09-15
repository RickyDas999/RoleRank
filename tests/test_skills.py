from __future__ import annotations

import pytest

from swetrack.domains.skills.normalization import get_skill_by_id, normalize_skill
from swetrack.domains.skills.taxonomy import TAXONOMY


def test_taxonomy_ids_and_slugs_are_unique():
    ids = [skill.id for skill in TAXONOMY]
    slugs = [skill.slug for skill in TAXONOMY]
    assert len(ids) == len(set(ids))
    assert len(slugs) == len(set(slugs))


def test_taxonomy_is_nonempty_and_covers_expected_categories():
    categories = {skill.category for skill in TAXONOMY}
    assert categories == {
        "programming_languages",
        "algorithms",
        "data_structures",
        "backend",
        "data_systems",
        "distributed_systems",
        "reliability",
        "system_design",
    }


@pytest.mark.parametrize("raw", ["Dynamic Programming", "dynamic programming", "DP", "  dp  ", "Dp"])
def test_normalize_dynamic_programming_aliases(raw):
    skill = normalize_skill(raw)
    assert skill is not None
    assert skill.id == "dynamic-programming"


@pytest.mark.parametrize("raw", ["DynamoDB", "Amazon DynamoDB", "AWS DynamoDB", "dynamodb"])
def test_normalize_dynamodb_variants_map_to_one_canonical_skill(raw):
    skill = normalize_skill(raw)
    assert skill is not None
    assert skill.id == "nosql"


def test_normalize_is_case_and_whitespace_insensitive():
    assert normalize_skill("  PYTHON ") is not None
    assert normalize_skill("  PYTHON ").id == "python"
    assert normalize_skill("python") == normalize_skill("Python")


def test_normalize_matches_real_sample_job_terms():
    # Grounded in data/sample_jobs.csv, not invented terms.
    assert normalize_skill("Kafka").id == "messaging"
    assert normalize_skill("Monitoring").id == "observability"


def test_normalize_unknown_skill_returns_none():
    assert normalize_skill("Quantum Computing") is None
    assert normalize_skill("") is None


def test_get_skill_by_id_known_and_unknown():
    skill = get_skill_by_id("python")
    assert skill is not None
    assert skill.name == "Python"
    assert get_skill_by_id("does-not-exist") is None
