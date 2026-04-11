"""
RAG retriever — semantic search interface for agents.

Provides high-level retrieval methods that combine vector similarity
search with graph context. Agents use this to find relevant prior
work before making their own contributions.
"""

import time
from typing import Any

from src.graph.repository import GraphRepository
from src.rag.store import VectorStore


class RAGRetriever:
    """
    High-level retrieval combining semantic search and graph context.

    Agents call these methods to gather context before reasoning.
    Results include both the text content and graph metadata
    (who created it, what it links to, current status).

    Usage:
        retriever = RAGRetriever(vector_store, graph_repo)
        context = await retriever.retrieve_for_inquiry(
            "antibiotic resistance mechanisms",
            include_types=["source_chunk", "note", "hypothesis"],
        )
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph: GraphRepository,
    ) -> None:
        self._store = vector_store
        self._graph = graph

    async def retrieve_for_inquiry(
        self,
        query: str,
        include_types: list[str] | None = None,
        top_k: int = 15,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant artifacts for a research inquiry.

        Searches across multiple artifact types and enriches results
        with graph metadata (status, confidence, creator).

        Args:
            query: Natural language query describing what to find.
            include_types: Artifact types to search. Defaults to all.
            top_k: Max results to return.

        Returns:
            List of enriched result dicts with text, score, and
            graph metadata.
        """
        default_types = [
            "source_chunk", "note", "hypothesis",
            "assumption", "finding",
        ]
        types = include_types or default_types

        metadata_filters = {"session_id": session_id} if session_id else None
        results = await self._store.search_by_type(
            query,
            types,
            top_k,
            metadata_filters=metadata_filters,
        )

        # Enrich with graph metadata
        enriched = []
        for r in results:
            node = await self._graph.get(r["artifact_id"])
            if node:
                r["status"] = node.get("status", "active")
                r["confidence"] = node.get("confidence", 0.5)
                r["created_by"] = node.get("created_by", "")
                r["labels"] = node.get("_labels", [])
            enriched.append(r)

        return enriched

    async def retrieve_source_chunks(
        self,
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve only source chunks relevant to a query."""
        return await self._store.search(
            query,
            top_k=top_k,
            artifact_type="source_chunk",
            metadata_filters={"session_id": session_id} if session_id else None,
        )

    async def retrieve_hypotheses(
        self,
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve hypotheses relevant to a query."""
        return await self._store.search(
            query,
            top_k=top_k,
            artifact_type="hypothesis",
            metadata_filters={"session_id": session_id} if session_id else None,
        )

    async def retrieve_notes(
        self,
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve notes relevant to a query."""
        return await self._store.search(
            query,
            top_k=top_k,
            artifact_type="note",
            metadata_filters={"session_id": session_id} if session_id else None,
        )

    async def find_similar_hypotheses(
        self,
        text: str,
        threshold: float = 0.85,
        exclude_agent: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find existing hypotheses semantically similar to a proposed one.

        Used for deduplication: before creating a new hypothesis, check
        if a near-identical one already exists. If similarity > threshold,
        the squid should create an EXTENDS relation instead of a
        duplicate.

        Args:
            text: The proposed hypothesis text to check against.
            threshold: Minimum cosine similarity (0.0–1.0) to consider
                      a match. 0.85 catches paraphrases while avoiding
                      false positives.
            exclude_agent: If set, exclude hypotheses created by this
                          agent (don't dedup against your own work).

        Returns:
            List of matching hypothesis dicts with artifact_id, text,
            score, and created_by. Sorted by similarity descending.
        """
        # Search for similar hypotheses in the vector store
        results = await self._store.search(
            text,
            top_k=5,
            artifact_type="hypothesis",
            metadata_filters={"session_id": session_id} if session_id else None,
        )

        # Filter by similarity threshold and optionally by agent
        matches = []
        for r in results:
            if r.get("score", 0) < threshold:
                continue

            # Enrich with graph data to check created_by
            node = await self._graph.get(r["artifact_id"])
            if not node:
                continue

            # Skip own hypotheses if requested
            if exclude_agent and node.get("created_by") == exclude_agent:
                continue

            # Skip non-active hypotheses
            if node.get("status") not in ("active", None):
                continue

            r["created_by"] = node.get("created_by", "")
            r["confidence"] = node.get("confidence", 0.5)
            matches.append(r)

        return matches

    async def retrieve_agent_context(
        self,
        agent_id: str,
        query: str,
        top_k: int = 20,
        session_id: str | None = None,
        graph_queries: Any | None = None,
    ) -> dict[str, list[dict]]:
        """
        Build a comprehensive context package for an agent.

        Combines graph traversal with vector similarity:
        1. Semantic search across all artifact types (including experiment_result)
        2. Boost agent's own work (sorted first within each type)
        3. Graph traversal: fetch experiment results + findings linked to
           the agent's hypotheses via TESTED_BY → PRODUCED chains

        Returns:
            Dict keyed by artifact type, each containing a list of relevant artifacts.
        """
        all_results = await self.retrieve_for_inquiry(
            query,
            include_types=[
                "source_chunk", "note", "hypothesis",
                "assumption", "finding", "experiment_result",
            ],
            top_k=top_k,
            session_id=session_id,
        )

        # Group by type, boosting agent's own artifacts to the front
        grouped: dict[str, list[dict]] = {}
        for r in all_results:
            atype = r.get("artifact_type", "unknown")
            if atype not in grouped:
                grouped[atype] = []
            grouped[atype].append(r)

        # Boost agent's own work — sort own artifacts first
        for atype in grouped:
            grouped[atype].sort(
                key=lambda r: (0 if r.get("created_by") == agent_id else 1),
            )

        # Graph traversal — fetch experiment results for agent's hypotheses
        if graph_queries:
            try:
                agent_hyps = await graph_queries.get_all_hypotheses(
                    created_by=agent_id, session_id=session_id
                )
                graph_results: list[dict] = []
                for h in agent_hyps[:10]:  # Cap at 10 hypotheses
                    hyp_context = await graph_queries.get_hypothesis_context(h["id"])
                    # Experiments + results from graph traversal
                    for exp_entry in hyp_context.get("experiments", []):
                        result = exp_entry.get("result")
                        if result:
                            graph_results.append({
                                **result,
                                "artifact_type": "experiment_result",
                                "score": 1.0,  # Graph results are exact matches
                                "source": "graph_traversal",
                            })
                    # Findings from graph traversal
                    for f in hyp_context.get("findings", []):
                        graph_results.append({
                            **f,
                            "artifact_type": "finding",
                            "score": 1.0,
                            "source": "graph_traversal",
                        })

                # Deduplicate graph results against vector results
                existing_ids = {
                    r.get("artifact_id") or r.get("id")
                    for rlist in grouped.values()
                    for r in rlist
                }
                for gr in graph_results:
                    gr_id = gr.get("artifact_id") or gr.get("id")
                    if gr_id and gr_id not in existing_ids:
                        atype = gr["artifact_type"]
                        if atype not in grouped:
                            grouped[atype] = []
                        grouped[atype].insert(0, gr)  # Prepend graph results
                        existing_ids.add(gr_id)
            except Exception:
                pass  # Graceful degradation — graph enrichment is optional

        return grouped

