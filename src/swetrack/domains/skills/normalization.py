"""Deterministic alias-based normalization of free-text skill strings.

Matching is case-insensitive, whitespace-normalized exact string matching
against each skill's name, slug, and aliases -- intentionally not fuzzy or
embedding-based (CLAUDE.md Phase 4: "Do not use embeddings merely because
they are available if deterministic normalization is sufficient").

The alias lookup is built once at import time and raises immediately if two
different skills claim the same alias, so taxonomy authoring mistakes fail
loudly instead of silently picking one match.
"""

from __future__ import annotations

from swetrack.domains.skills.models import Skill
from swetrack.domains.skills.taxonomy import TAXONOMY


def _normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _build_lookup(taxonomy: list[Skill]) -> dict[str, Skill]:
    lookup: dict[str, Skill] = {}
    for skill in taxonomy:
        for key in (skill.name, skill.slug, *skill.aliases):
            normalized = _normalize_key(key)
            existing = lookup.get(normalized)
            if existing is not None and existing.id != skill.id:
                raise ValueError(
                    f"Ambiguous skill alias {normalized!r} maps to both "
                    f"{existing.id!r} and {skill.id!r}"
                )
            lookup[normalized] = skill
    return lookup


_ALIAS_LOOKUP = _build_lookup(TAXONOMY)


def normalize_skill(raw: str) -> Skill | None:
    """Return the canonical Skill for a free-text skill string, or None if unrecognized."""
    return _ALIAS_LOOKUP.get(_normalize_key(raw))


def get_skill_by_id(skill_id: str) -> Skill | None:
    """Look up a canonical skill by its id/slug."""
    for skill in TAXONOMY:
        if skill.id == skill_id:
            return skill
    return None
