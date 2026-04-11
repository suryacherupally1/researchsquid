"""
Hindsight memory client — wraps the Hindsight Python SDK for ResearchSquid.

Provides a clean interface for the three memory operations:
- retain: Store structured facts from agent interactions
- recall: Multi-strategy retrieval (semantic + BM25 + graph + temporal)
- reflect: Synthesize new knowledge from accumulated memories
"""

import logging
from typing import Any

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Per-agent memory interface backed by Hindsight.

    Each instance manages a private bank for the agent and
    read access to the shared institutional bank.

    Usage:
        memory = AgentMemory(agent_id, session_id, hindsight_client, config)

        # After each research cycle — retain what happened
        await memory.retain_iteration(iteration, output, experiment_results)

        # Before reasoning — recall relevant context
        context = await memory.recall_for_research(query)

        # At end of cycle — reflect on progress
        insight = await memory.reflect_on_progress(subproblem)
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        client: Any,
        config: Settings | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._client = client
        self._config = config or default_settings
        self._private_bank = f"squid-{agent_id}-{session_id}"
        self._shared_bank = f"institute-{session_id}"

    @property
    def private_bank_id(self) -> str:
        return self._private_bank

    @property
    def shared_bank_id(self) -> str:
        return self._shared_bank

    # ── RETAIN ────────────────────────────────────────────────────────

    async def retain_iteration(
        self,
        iteration: int,
        findings_summary: str,
        hypotheses: list[dict[str, Any]],
        experiment_results: list[dict[str, Any]] | None = None,
        opencode_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Retain structured facts from a research iteration.

        Called at the end of SquidAgent.run() — replaces append_memory().

        Hindsight auto-extracts:
        - Entities (hypothesis names, experiment IDs, confidence values)
        - Relationships (H1 tested by E3, E3 supports H1)
        - Temporal data (iteration number, timestamp)

        Facts are deduplicated across iterations, entities are linked
        in a graph for traversal during recall, and temporal metadata
        enables "what changed since iteration 2?" queries.
        """
        content = self._format_iteration_experience(
            iteration, findings_summary, hypotheses
        )
        await self._client.aretain(
            bank_id=self._private_bank,
            content=content,
            context=f"research-iteration-{iteration}",
        )

        if experiment_results:
            for result in experiment_results:
                await self._client.aretain(
                    bank_id=self._private_bank,
                    content=self._format_experiment_experience(result),
                    context=f"experiment-result-iteration-{iteration}",
                )

        if opencode_results:
            for oc_result in opencode_results:
                await self._client.aretain(
                    bank_id=self._private_bank,
                    content=self._format_opencode_experience(oc_result),
                    context=f"opencode-result-iteration-{iteration}",
                )

    async def retain_debate_outcome(
        self,
        iteration: int,
        hypothesis_id: str,
        outcome: str,
        counter_arguments: list[str],
    ) -> None:
        """
        Retain the outcome of a debate round related to the agent's hypothesis.

        Creates an experience record that captures what was challenged,
        what counter-arguments were made, and whether the hypothesis
        survived, was modified, or was refuted. This allows agents to
        learn from debate patterns across iterations.
        """
        content = (
            f"Debate outcome for hypothesis {hypothesis_id[:12]}:\n"
            f"Result: {outcome}\n"
            f"Counter-arguments received:\n"
            + "\n".join(f"- {ca[:200]}" for ca in counter_arguments[:5])
        )
        await self._client.aretain(
            bank_id=self._private_bank,
            content=content,
            context=f"debate-outcome-iteration-{iteration}",
        )

    async def retain_to_shared(self, content: str, context: str) -> None:
        """
        Retain information to the shared institutional bank.

        Used for iteration briefings, cross-agent discoveries,
        high-confidence findings promoted from private banks,
        and debate consensus positions.
        """
        await self._client.aretain(
            bank_id=self._shared_bank,
            content=content,
            context=context,
        )

    # ── RECALL ────────────────────────────────────────────────────────

    async def recall_for_research(
        self,
        query: str,
        include_shared: bool = True,
    ) -> dict[str, Any]:
        """
        Recall relevant memories before a research cycle.

        Performs multi-strategy retrieval (semantic + BM25 + graph + temporal)
        across the agent's private bank and optionally the shared bank.

        Returns a structured context dict that can be injected into the
        squid's research prompt alongside the existing RAG context.
        """
        private_memories = await self._client.arecall(
            bank_id=self._private_bank,
            query=query,
        )

        result: dict[str, Any] = {
            "private_memories": private_memories,
            "shared_memories": None,
        }

        if include_shared:
            shared_memories = await self._client.arecall(
                bank_id=self._shared_bank,
                query=query,
            )
            result["shared_memories"] = shared_memories

        return result

    async def recall_past_failures(
        self,
        hypothesis_text: str,
    ) -> list[dict]:
        """
        Specifically recall failed approaches related to a hypothesis.

        Used before designing new experiments to prevent repeating
        failed approaches.
        """
        query = (
            f"Failed experiments, inconclusive results, or refuted approaches "
            f"related to: {hypothesis_text}"
        )
        return await self._client.arecall(
            bank_id=self._private_bank,
            query=query,
        )

    async def recall_iteration_history(
        self,
        from_iteration: int | None = None,
        to_iteration: int | None = None,
    ) -> list[dict]:
        """
        Recall memories from a specific iteration range.

        Leverages Hindsight's temporal filtering to answer questions like
        "What did I learn between iteration 2 and iteration 4?"
        """
        query = "Research progress and findings"
        if from_iteration is not None and to_iteration is not None:
            query += f" from iteration {from_iteration} to {to_iteration}"
        elif from_iteration is not None:
            query += f" since iteration {from_iteration}"

        return await self._client.arecall(
            bank_id=self._private_bank,
            query=query,
        )

    # ── REFLECT ───────────────────────────────────────────────────────

    async def reflect_on_progress(
        self,
        subproblem: str,
    ) -> str:
        """
        Synthesize a strategic assessment from all accumulated memories.

        Called at the end of each research cycle to generate a high-level
        reflection that goes into the agent's next iteration context.

        Instead of manually gathering context and asking the LLM to
        summarize, Hindsight's reflect operation retrieves ALL relevant
        memories, applies hierarchical priority (Mental Models >
        Observations > raw facts), and synthesizes a disposition-aware
        response.
        """
        return await self._client.areflect(
            bank_id=self._private_bank,
            query=(
                f"Based on all my experiments, findings, and debate outcomes so far, "
                f"what is the current state of my research on '{subproblem}'? "
                f"What are my strongest hypotheses? What approaches failed? "
                f"What should I investigate next?"
            ),
        )

    async def reflect_on_hypothesis(
        self,
        hypothesis_text: str,
        hypothesis_id: str,
    ) -> str:
        """
        Deep reflection on a specific hypothesis.

        Synthesizes all evidence (experiments, debate outcomes, confidence
        changes) into a coherent assessment of one hypothesis. Used before
        deciding whether to continue, modify, or abandon.
        """
        return await self._client.areflect(
            bank_id=self._private_bank,
            query=(
                f"What is all the evidence for and against hypothesis '{hypothesis_text}' "
                f"(ID: {hypothesis_id[:12]})? Including experiment results, debate challenges, "
                f"and confidence changes. Should I continue pursuing it?"
            ),
        )

    async def reflect_cross_agent(
        self,
        query: str,
    ) -> str:
        """
        Reflect over the shared institutional bank.

        Used by the controller and iteration briefing to synthesize
        cross-agent progress for the whole research session.
        """
        return await self._client.areflect(
            bank_id=self._shared_bank,
            query=query,
        )

    # ── Formatting helpers ────────────────────────────────────────────

    def _format_iteration_experience(
        self,
        iteration: int,
        findings_summary: str,
        hypotheses: list[dict[str, Any]],
    ) -> str:
        """Format an iteration into a structured experience for retain."""
        hyp_lines = []
        for h in hypotheses:
            hyp_lines.append(
                f"  - {h.get('statement', h.get('text', ''))[:100]} "
                f"(conf={h.get('confidence', 0):.2f}, "
                f"status={h.get('adjudication_status', 'pending')})"
            )
        return (
            f"Research iteration {iteration} completed.\n\n"
            f"Findings:\n{findings_summary}\n\n"
            f"Current hypotheses:\n" + "\n".join(hyp_lines)
        )

    def _format_experiment_experience(self, result: dict) -> str:
        """Format an experiment result into a structured experience."""
        return (
            f"Experiment {result.get('experiment_id', '?')[:12]} completed.\n"
            f"Exit code: {result.get('exit_code', '?')}\n"
            f"Stdout (first 300 chars): {result.get('stdout', '')[:300]}\n"
            f"Interpretation: {result.get('interpretation', 'None')}\n"
            f"Hypothesis tested: {result.get('hypothesis_id', '?')[:12]}"
        )

    def _format_opencode_experience(self, result: dict) -> str:
        """Format an OpenCode task result into a structured experience."""
        return (
            f"OpenCode task '{result.get('topic', '?')}' completed.\n"
            f"Status: {result.get('status', '?')}\n"
            f"Files produced: {', '.join(result.get('files_produced', []))}\n"
            f"Satisfied: {result.get('satisfied', False)}\n"
            f"Iterations: {result.get('total_iterations', 0)}"
        )
