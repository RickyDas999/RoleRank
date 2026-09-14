"""Learning domain: practice activities, attempts, and immutable SkillEvents.

Skill mastery is never stored as a single mutable score. Instead, every
practice attempt emits one or more immutable SkillEvent observations (see
CLAUDE.md Phase 6), so a future mastery model (Bayesian Knowledge Tracing,
M6) can replay the full history rather than trusting a running average.
"""
