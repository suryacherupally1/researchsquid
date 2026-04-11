"""
Hindsight-based memory layer for agent working memory.

This replaces the append-only memory.md + beliefs.json workspace files with
a structured memory system that supports retain (store), recall (search),
and reflect (synthesize) operations.

Architecture:
- Each squid gets a private Hindsight memory bank
- All agents share an institutional memory bank
- Neo4j remains the institutional knowledge graph (hypotheses, findings, experiments)
- pgvector remains for source chunk retrieval
- Hindsight handles per-agent working memory with temporal awareness
"""
