"""Canonical Skill schema shared across the taxonomy and normalization.

``id`` is the same value as ``slug`` for now -- a stable, human-readable
string identifier. A separate surrogate key would be premature without a
database behind this domain yet.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SkillCategory = Literal[
    "programming_languages",
    "algorithms",
    "data_structures",
    "backend",
    "data_systems",
    "distributed_systems",
    "reliability",
]


class Skill(BaseModel):
    """One canonical software-engineering skill node in the taxonomy."""

    id: str = Field(..., min_length=1)
    slug: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    category: SkillCategory
    parent_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
