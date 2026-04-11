"""
Squid agent — the core researcher of the institute.

Each Squid owns a line of inquiry (a subproblem from the Director).
It reads sources, writes notes, forms hypotheses, designs experiments,
and engages with other agents' work through relations and messages.

Each squid has a unique AgentPersona that shapes its behavior
through prompt injection and model selection. A skeptic reasons
differently from an empiricist, even on the same subproblem.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from src.config import Settings, settings as default_settings
from src.events.bus import EventBus
from src.graph.repository import GraphRepository
from src.llm.client import LLMClient
from src.llm.prompts import SQUID_SYSTEM, SQUID_ANALYZE
from src.models.agent_state import SquidState
from src.models.claim import Assumption, Hypothesis
from src.models.events import Event, EventType
from src.models.experiment import Experiment, ExperimentSpec
from src.models.message import Message, MessageType, MESSAGE_PRIORITY
from src.models.note import Note
from src.models.persona import AgentPersona, generate_persona_prompt
from src.models.relation import Relation, RelationType
from src.rag.indexer import RAGIndexer
from src.rag.retriever import RAGRetriever
from src.search.arxiv import ArxivSearch
from src.search.tavily import TavilySearch
from src.agents.workspace_tools import WorkspaceTools, OpenCodeTask

logger = logging.getLogger(__name__)


class SquidOutput(BaseModel):
    """Structured output from a Squid's analysis cycle."""

    notes: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    experiment_proposals: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    search_queries: list[dict[str, Any]] = Field(default_factory=list)
    opencode_task: OpenCodeTask | None = Field(
        default=None,
        description=(
            "Optional code exploration task to delegate to OpenCode. "
            "Only set when code analysis would materially advance a hypothesis."
        ),
    )


class SquidAgent:
    """
    A research squid that investigates a specific subproblem.

    Each squid cycle:
    1. Retrieves relevant context (RAG + graph)
    2. Checks for unread messages from other agents
    3. Analyzes source material
    4. Produces notes, assumptions, hypotheses
    5. Creates relations to existing work
    6. Proposes experiments
    7. Sends messages to other agents if needed
    8. Optionally searches for more sources (Tavily/Arxiv)

    Multiple squids run in parallel via LangGraph Send.
    """

    def __init__(
        self,
        llm: LLMClient,
        graph: GraphRepository,
        retriever: RAGRetriever,
        indexer: RAGIndexer | None,
        event_bus: EventBus,
        tavily: TavilySearch | None = None,
        arxiv_search: ArxivSearch | None = None,
        config: Settings | None = None,
        workspace_tools: WorkspaceTools | None = None,
        graph_queries: Any | None = None,
        agent_memory: Any | None = None,
    ) -> None:
        self._llm = llm
        self._graph = graph
        self._retriever = retriever
        self._indexer = indexer
        self._bus = event_bus
        self._tavily = tavily
        self._arxiv = arxiv_search
        self._config = config or default_settings
        self._workspace = workspace_tools
        self._graph_queries = graph_queries
        self._memory = agent_memory
        self._source_ingest_locks: dict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    async def run(self, state: SquidState) -> dict[str, Any]:
        """
        Execute one research cycle for this squid.

        The persona (if present) shapes behavior in two ways:
        1. Prompt injection: behavioral traits become LLM instructions
        2. Model selection: persona.model_tier picks the LLM model

        Args:
            state: The squid's current state with subproblem assignment.

        Returns:
            State update dict with IDs of all created artifacts.
        """
        squid_start = time.time()
        agent_id = state["agent_id"]
        agent_name = state["agent_name"]
        subproblem = state["subproblem"]
        query = subproblem["question"]
        session_id = state.get("session_id", "")

        # Reconstruct persona from serialized state (if present)
        persona_dict = state.get("persona", {})
        persona = AgentPersona(**persona_dict) if persona_dict else None

        await self._bus.publish(Event(
            event_type=EventType.AGENT_THINKING,
            agent_id=agent_id,
            payload={
                "inquiry": query,
                "archetype": persona.archetype_id if persona else None,
                "model_tier": persona.model_tier if persona else "default",
            },
        ))

        # 1. Retrieve context from RAG
        rag_start = time.time()
        context = await self._retriever.retrieve_agent_context(
            agent_id,
            query,
            top_k=self._config.retrieval_agent_context_top_k,
            session_id=session_id,
            graph_queries=self._graph_queries,
        )
        #print(f"[TIMING] Squid {agent_id[:12]} RAG retrieval: {time.time() - rag_start:.2f}s")

        # 2. Check for unread messages
        msg_start = time.time()
        unread = await self._graph.get_unread_messages(agent_id)
        for msg in unread:
            await self._graph.mark_message_read(msg["id"])
        #print(f"[TIMING] Squid {agent_id[:12]} message check: {time.time() - msg_start:.2f}s")

        # 3. Format context for the prompt
        source_chunks = self._format_artifacts(context.get("source_chunk", []))
        existing_work = self._format_existing_work(context)
        messages_text = self._format_messages(unread)

        # 3b. Hindsight memory recall — augments RAG with agent's working memory
        memory_section = ""
        if self._memory:
            mem_context = await self._memory.recall_for_research(query)
            private = mem_context.get("private_memories")
            if private:
                memory_section = self._format_memory_context(private)

        # 4. Build system prompt with persona injection
        system_prompt = SQUID_SYSTEM
        if persona:
            persona_block = generate_persona_prompt(persona, config=self._config)
            system_prompt = f"{SQUID_SYSTEM}\n\n{persona_block}"

        # 5. Include institute briefing if available
        briefing = state.get("iteration_summary", "")
        briefing_section = (
            f"\n\n=== INSTITUTE BRIEFING ===\n{briefing}"
            if briefing else ""
        )

        # 6. Call LLM for analysis
        llm_start = time.time()
        prompt = SQUID_ANALYZE.format(
            agent_name=agent_name,
            agent_id=agent_id,
            line_of_inquiry=query,
            source_chunks=source_chunks or "No source material found yet.",
            existing_work=existing_work or "No existing work from other agents.",
            messages=messages_text or "No messages.",
        ) + briefing_section + memory_section

        # Define usage tracking for this specific execution
        usage_accumulator: dict[str, Any] = {"cost": 0.0}

        # Use persona-aware model selection if persona exists
        if persona:
            output = await self._llm.complete_structured_for_persona(
                prompt=prompt,
                response_model=SquidOutput,
                persona=persona,
                system=system_prompt,
                temperature=self._config.temperature_squid,
                usage_accumulator=usage_accumulator,
            )
        else:
            output = await self._llm.complete_structured(
                prompt=prompt,
                response_model=SquidOutput,
                system=system_prompt,
                temperature=self._config.temperature_squid,
                usage_accumulator=usage_accumulator,
            )
        #print(f"[TIMING] Squid {agent_id[:12]} LLM call: {time.time() - llm_start:.2f}s")

        logger.warning(
            f"Squid {agent_id[:12]} LLM output: "
            f"notes={len(output.notes)}, "
            f"hypotheses={len(output.hypotheses)}, "
            f"assumptions={len(output.assumptions)}, "
            f"search_queries={len(output.search_queries)}, "
            f"experiments={len(output.experiment_proposals)}, "
            f"messages={len(output.messages)}, "
            f"cost=${usage_accumulator.get('cost', 0.0):.6f}"
        )

        # 5. Store all artifacts in the graph
        store_start = time.time()
        result = await self._store_artifacts(output, agent_id, state)
        # Attach the precise agent iteration cost
        result["spent_usd"] = usage_accumulator.get("cost", 0.0)
        #print(f"[TIMING] Squid {agent_id[:12]} store artifacts: {time.time() - store_start:.2f}s")

        # 5b. Workspace updates — only if workspace layer is active
        if self._workspace:
            iteration = state.get("iteration", 0)
            # Fetch the agent's current hypotheses from graph for workspace sync
            agent_hypotheses = await self._graph.get_by_label(
                "Hypothesis",
                filters={"created_by": agent_id, "session_id": session_id},
                limit=100,
            )
            findings_summary = self._summarize_iteration_for_memory(output)
            await self._workspace.append_memory(findings_summary, iteration)
            await self._workspace.sync_hypotheses_from_dag(agent_hypotheses)
            await self._workspace.update_beliefs(agent_hypotheses)
            if output.opencode_task:
                loop_result = await self._workspace.run_opencode_loop(
                    output.opencode_task
                )
                if loop_result:
                    status = "satisfied" if loop_result.satisfied else "not satisfied"
                    await self._workspace.append_memory(
                        f"- OpenCode task '{output.opencode_task.topic}': "
                        f"{loop_result.total_iterations} iterations, {status}, "
                        f"produced: {', '.join(loop_result.files_produced) or 'no files'}",
                        iteration,
                    )

        # 6. Execute search queries if any
        search_start = time.time()
        await self._execute_searches(
            output.search_queries,
            agent_id,
            session_id,
        )
        # if output.search_queries:
            #print(f"[TIMING] Squid {agent_id[:12]} searches: {time.time() - search_start:.2f}s")

        # 7. Retain this iteration's work in Hindsight memory
        if self._memory:
            iteration = state.get("iteration", 0)
            findings_summary = self._summarize_iteration_for_memory(output)
            agent_hypotheses = await self._graph.get_by_label(
                "Hypothesis",
                filters={"created_by": agent_id, "session_id": session_id},
                limit=20,
            )
            await self._memory.retain_iteration(
                iteration=iteration,
                findings_summary=findings_summary,
                hypotheses=agent_hypotheses,
            )

        #print(f"[TIMING] Squid {agent_id[:12]} TOTAL: {time.time() - squid_start:.2f}s")

        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id=agent_id,
            payload={
                "notes": len(result.get("notes_created", [])),
                "hypotheses": len(result.get("hypotheses_created", [])),
                "relations": len(result.get("relations_created", [])),
                "experiments": len(result.get("experiments_proposed", [])),
            },
        ))

        return result

    async def _store_artifacts(
        self,
        output: SquidOutput,
        agent_id: str,
        state: SquidState,
    ) -> dict[str, Any]:
        """Store all LLM-generated artifacts in the knowledge graph."""
        notes_created: list[str] = []
        assumptions_created: list[str] = []
        hypotheses_created: list[str] = []
        relations_created: list[str] = []
        experiments_proposed: list[str] = []
        messages_sent: list[str] = []

        # Store notes
        for note_data in output.notes:
            note = Note(
                text=note_data.get("text", ""),
                source_chunk_ids=note_data.get("source_chunk_ids", []),
                created_by=agent_id,
                confidence=note_data.get(
                    "confidence", self._config.note_default_confidence
                ),
            )
            await self._graph.create(note)
            if note.source_chunk_ids:
                await self._graph.link_note_to_chunks(
                    note.id, note.source_chunk_ids
                )
            notes_created.append(note.id)

        # Store assumptions
        for assum_data in output.assumptions:
            assumption = Assumption(
                text=assum_data.get("text", ""),
                basis=assum_data.get("basis", ""),
                strength=assum_data.get("strength", "moderate"),
                created_by=agent_id,
            )
            await self._graph.create(assumption)
            assumptions_created.append(assumption.id)

        # Store hypotheses (with deduplication)
        for hyp_data in output.hypotheses:
            hyp_text = hyp_data.get("text", "")
            if not hyp_text:
                continue

            # Check for near-duplicate hypotheses from other agents
            similar = await self._retriever.find_similar_hypotheses(
                hyp_text,
                threshold=self._config.hypothesis_dedup_threshold,
                exclude_agent=agent_id,
                session_id=state.get("session_id", ""),
            )

            if similar:
                # Near-duplicate found — link to existing instead of creating
                existing_id = similar[0]["artifact_id"]
                relation = Relation(
                    source_artifact_id=existing_id,
                    target_artifact_id=existing_id,
                    relation_type=RelationType.EXTENDS,
                    reasoning=(
                        f"Agent {agent_id} independently proposed a similar "
                        f"hypothesis (dedup merged)"
                    ),
                    weight=self._config.dedup_relation_weight,
                    created_by=agent_id,
                )
                await self._graph.create_relation(relation)
                # Still track the existing hypothesis as "ours" for context
                hypotheses_created.append(existing_id)
                continue

            hypothesis = Hypothesis(
                text=hyp_text,
                supporting_evidence=hyp_data.get("supporting_evidence", []),
                testable=hyp_data.get("testable", True),
                created_by=agent_id,
                confidence=hyp_data.get(
                    "confidence", self._config.hypothesis_default_confidence
                ),
            )
            await self._graph.create(hypothesis)
            hypotheses_created.append(hypothesis.id)

        # Store relations
        for rel_data in output.relations:
            relation = Relation(
                source_artifact_id=rel_data.get("source_artifact_id", ""),
                target_artifact_id=rel_data.get("target_artifact_id", ""),
                relation_type=RelationType.from_llm(
                    rel_data.get("relation_type", "")
                ),
                reasoning=rel_data.get("reasoning", ""),
                weight=rel_data.get(
                    "weight", self._config.relation_default_weight
                ),
                created_by=agent_id,
            )
            if relation.source_artifact_id and relation.target_artifact_id:
                await self._graph.create_relation(relation)
                relations_created.append(relation.id)

        # Store experiment proposals
        for exp_data in output.experiment_proposals:
            spec = ExperimentSpec(
                code=exp_data.get("code", ""),
                expected_outcome=exp_data.get("expected_outcome", ""),
                timeout_seconds=exp_data.get(
                    "timeout_seconds",
                    self._config.default_experiment_timeout_seconds,
                ),
            )
            experiment = Experiment(
                hypothesis_id=exp_data.get("hypothesis_id", ""),
                spec=spec,
                created_by=agent_id,
            )
            await self._graph.create(experiment)
            if experiment.hypothesis_id:
                await self._graph.link_hypothesis_to_experiment(
                    experiment.hypothesis_id, experiment.id
                )
            experiments_proposed.append(experiment.id)

        # Send messages (with typed protocol)
        for msg_data in output.messages:
            message = Message(
                from_agent=agent_id,
                to_agent=msg_data.get("to_agent", ""),
                text=msg_data.get("text", ""),
                message_type=MessageType.from_llm(
                    msg_data.get("message_type", "question")
                ),
                regarding_artifact_id=msg_data.get("regarding_artifact_id", ""),
                created_by=agent_id,
            )
            if message.to_agent:
                await self._graph.create_message(message)
                messages_sent.append(message.id)

        return {
            "notes_created": notes_created,
            "assumptions_created": assumptions_created,
            "hypotheses_created": hypotheses_created,
            "relations_created": relations_created,
            "experiments_proposed": experiments_proposed,
            "messages_sent": messages_sent,
        }

    async def _execute_searches(
        self,
        queries: list[dict[str, Any]],
        agent_id: str,
        session_id: str,
    ) -> None:
        """Execute any search queries the squid requested."""
        for q in queries:
            source = q.get("source", "tavily")
            query = q.get("query", "")

            if not query:
                continue

            if source == "arxiv" and self._arxiv:
                papers = await self._arxiv.search(
                    query,
                    max_results=self._config.squid_search_max_results,
                    agent_id=agent_id,
                )
                if self._indexer:
                    await self._ingest_arxiv_results(
                        papers,
                        agent_id=agent_id,
                        session_id=session_id,
                    )
            elif source == "tavily" and self._tavily:
                await self._tavily.search(
                    query,
                    max_results=self._config.squid_search_max_results,
                    agent_id=agent_id,
                )

    async def _ingest_arxiv_results(
        self,
        papers: list[dict[str, Any]],
        agent_id: str,
        session_id: str,
    ) -> None:
        """Download and ingest discovered arXiv papers into shared memory in parallel."""
        if not self._indexer or not papers:
            return

        tasks = [
            self._ingest_single_arxiv_paper(paper, agent_id, session_id)
            for paper in papers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _ingest_single_arxiv_paper(
        self,
        paper: dict[str, Any],
        agent_id: str,
        session_id: str,
    ) -> None:
        """Worker to download and ingest a single arXiv paper."""
        arxiv_id = str(paper.get("arxiv_id", "")).strip()
        if not arxiv_id:
            return

        canonical_uri = f"arxiv:{arxiv_id}"
        async with self._source_ingest_locks[canonical_uri]:
            existing = await self._graph.get_by_label(
                "Source",
                filters={
                    "uri": canonical_uri,
                    "session_id": session_id,
                },
                limit=1,
            )
            if existing:
                await self._bus.publish(Event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id=agent_id,
                    payload={
                        "action": "search_source_already_ingested",
                        "source": "arxiv",
                        "title": paper.get("title", ""),
                        "arxiv_id": arxiv_id,
                        "source_id": existing[0].get("id", ""),
                    },
                ))
                return

            await self._bus.publish(Event(
                event_type=EventType.AGENT_ACTION,
                agent_id=agent_id,
                payload={
                    "action": "downloading_source",
                    "source": "arxiv",
                    "title": paper.get("title", ""),
                    "arxiv_id": arxiv_id,
                },
            ))

            try:
                pdf_path = await self._arxiv.download(
                    arxiv_id,
                    agent_id=agent_id,
                )
                await self._bus.publish(Event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id=agent_id,
                    payload={
                        "action": "ingesting_source",
                        "source": "arxiv",
                        "title": paper.get("title", ""),
                        "arxiv_id": arxiv_id,
                        "progress": 100,
                        "stage": "ingesting",
                    },
                ))
                source_id = await self._indexer.ingest_pdf(pdf_path, agent_id)
                await self._graph.update(
                    source_id,
                    {
                        "source_type": "arxiv",
                        "uri": canonical_uri,
                        "title": paper.get("title", ""),
                        "file_path": pdf_path,
                    },
                )
                await self._bus.publish(Event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id=agent_id,
                    payload={
                        "action": "ingested_search_source",
                        "source": "arxiv",
                        "title": paper.get("title", ""),
                        "arxiv_id": arxiv_id,
                        "source_id": source_id,
                        "file_path": pdf_path,
                    },
                ))
            except Exception as exc:
                await self._bus.publish(Event(
                    event_type=EventType.ERROR,
                    agent_id=agent_id,
                    payload={
                        "error": (
                            f"Failed to download or ingest arXiv paper "
                            f"{arxiv_id}: {exc}"
                        ),
                    },
                ))

    def _summarize_iteration_for_memory(self, output: SquidOutput) -> str:
        """
        Format iteration output as a memory.md entry.

        Produces structured bullet points from notes/hypotheses created.
        Does NOT call LLM — just formats the already-produced data.
        """
        lines: list[str] = []
        if output.notes:
            lines.append(f"- Wrote {len(output.notes)} notes")
            for n in output.notes[:3]:  # First 3 to keep memory concise
                lines.append(f"  - {n.get('text', '')[:100]}")
        if output.hypotheses:
            lines.append(f"- Formed {len(output.hypotheses)} hypotheses")
        if output.experiment_proposals:
            lines.append(f"- Proposed {len(output.experiment_proposals)} experiments")
        if not lines:
            lines.append("- No significant artifacts produced this cycle")
        return "\n".join(lines)

    def _format_artifacts(self, artifacts: list[dict]) -> str:
        """Format a list of artifact dicts into readable text for prompts."""
        if not artifacts:
            return ""
        parts = []
        for a in artifacts:
            parts.append(
                f"[{a.get('artifact_type', 'unknown')}] "
                f"(ID: {a.get('artifact_id', '?')}, "
                f"confidence: {a.get('confidence', '?')}, "
                f"by: {a.get('created_by', '?')})\n"
                f"{a.get('text', '')}\n"
            )
        return "\n---\n".join(parts)

    def _format_existing_work(self, context: dict[str, list[dict]]) -> str:
        """Format all existing work (excluding source chunks) for the prompt."""
        parts = []
        for atype in ["note", "hypothesis", "assumption", "finding", "experiment_result"]:
            artifacts = context.get(atype, [])
            if artifacts:
                parts.append(f"\n=== {atype.upper()}S ===")
                if atype == "experiment_result":
                    parts.append(self._format_experiment_results(artifacts))
                else:
                    parts.append(self._format_artifacts(artifacts))
        return "\n".join(parts)

    def _format_experiment_results(self, results: list[dict]) -> str:
        """Format experiment results for the prompt — emphasis on what was tested and what happened."""
        parts = []
        for r in results[:10]:  # Cap at 10 to avoid prompt bloat
            parts.append(
                f"[experiment_result] (exit_code: {r.get('exit_code', '?')}, "
                f"experiment: {(r.get('experiment_id', '') or r.get('id', '?'))[:12]})\n"
                f"Stdout: {(r.get('stdout', '') or '')[:300]}\n"
                f"Interpretation: {r.get('interpretation', 'None')}"
            )
        return "\n---\n".join(parts)

    def _format_messages(self, messages: list[dict]) -> str:
        """
        Format unread messages for the prompt, grouped by type.

        Messages are sorted by priority: dependency warnings and
        objections first, acknowledgments last. This ensures the
        agent addresses critical feedback before routine messages.
        """
        if not messages:
            return ""

        # Sort by message type priority (lower = more urgent)
        def msg_priority(m: dict) -> int:
            mtype = m.get("message_type", "question")
            try:
                return MESSAGE_PRIORITY.get(MessageType(mtype), 4)
            except ValueError:
                return 4

        sorted_msgs = sorted(messages, key=msg_priority)

        parts = []
        current_type = None
        for m in sorted_msgs:
            mtype = m.get("message_type", "question").upper()
            if mtype != current_type:
                current_type = mtype
                parts.append(f"\n--- {current_type} ---")

            parts.append(
                f"From {m.get('from_agent', '?')} [{mtype}]: "
                f"{m.get('text', '')}"
                f" (re: {m.get('regarding_artifact_id', 'general')})"
            )
        return "\n".join(parts)

    def _format_memory_context(self, memories: list[dict] | Any) -> str:
        """Format Hindsight recall results into a prompt section."""
        if not memories:
            return ""
        parts = ["\n\n=== AGENT WORKING MEMORY ==="]
        if isinstance(memories, list):
            for m in memories[:8]:
                text = m.get("content", m.get("text", str(m)))[:300]
                parts.append(f"- {text}")
        elif isinstance(memories, str):
            parts.append(memories[:2000])
        else:
            parts.append(str(memories)[:2000])
        return "\n".join(parts)

