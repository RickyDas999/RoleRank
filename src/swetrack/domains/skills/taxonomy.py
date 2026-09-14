"""Initial canonical software-engineering skill taxonomy.

A small, deliberately incomplete starting set matching the example taxonomy
in CLAUDE.md Phase 4 -- hand-curated and deterministic, not derived from any
external ontology or embedding model. ``parent_id`` is left unset on every
entry: there are no subcategories yet, so ``category`` alone captures the
grouping shown in CLAUDE.md's tree.

Aliases are intentionally sparse and only added where a mapping is either
explicitly called out in CLAUDE.md (Dynamic Programming/DP,
DynamoDB/Amazon DynamoDB/AWS DynamoDB -> NoSQL) or grounded in terms that
actually appear in data/sample_jobs.csv (Kafka -> Messaging,
Monitoring -> Observability).
"""

from __future__ import annotations

from swetrack.domains.skills.models import Skill, SkillCategory


def _skill(slug: str, name: str, category: SkillCategory, aliases: list[str] | None = None) -> Skill:
    return Skill(id=slug, slug=slug, name=name, category=category, aliases=aliases or [])


TAXONOMY: list[Skill] = [
    # Programming Languages
    _skill("python", "Python", "programming_languages", ["py"]),
    _skill("java", "Java", "programming_languages"),
    _skill("javascript", "JavaScript", "programming_languages", ["js"]),
    # Algorithms
    _skill("dynamic-programming", "Dynamic Programming", "algorithms", ["dp"]),
    _skill("graphs", "Graphs", "algorithms", ["graph theory", "graph algorithms"]),
    _skill("trees", "Trees", "algorithms", ["binary trees"]),
    _skill("binary-search", "Binary Search", "algorithms"),
    # Data Structures
    _skill("hash-maps", "Hash Maps", "data_structures", ["hash tables", "hashmaps", "dictionaries"]),
    _skill("heaps", "Heaps", "data_structures", ["priority queue", "priority queues"]),
    _skill("queues", "Queues", "data_structures"),
    _skill("tries", "Tries", "data_structures", ["trie", "prefix tree"]),
    # Backend
    _skill("rest-apis", "REST APIs", "backend", ["rest api", "restful api", "restful apis"]),
    _skill("authentication", "Authentication", "backend", ["auth", "oauth", "jwt"]),
    _skill("async-processing", "Async Processing", "backend", ["asynchronous processing", "asyncio"]),
    # Data Systems
    _skill("sql", "SQL", "data_systems", ["postgresql", "postgres", "mysql"]),
    _skill("nosql", "NoSQL", "data_systems", ["dynamodb", "amazon dynamodb", "aws dynamodb", "mongodb"]),
    _skill("indexing", "Indexing", "data_systems"),
    _skill("data-modeling", "Data Modeling", "data_systems"),
    # Distributed Systems
    _skill("caching", "Caching", "distributed_systems", ["redis", "memcached"]),
    _skill("partitioning", "Partitioning", "distributed_systems", ["sharding"]),
    _skill("replication", "Replication", "distributed_systems"),
    _skill("messaging", "Messaging", "distributed_systems", ["kafka", "message queue", "message queues", "pub/sub"]),
    _skill("consistency", "Consistency", "distributed_systems"),
    # Reliability
    _skill("retries", "Retries", "reliability"),
    _skill("idempotency", "Idempotency", "reliability"),
    _skill("failover", "Failover", "reliability"),
    _skill("observability", "Observability", "reliability", ["monitoring", "logging", "tracing"]),
]
