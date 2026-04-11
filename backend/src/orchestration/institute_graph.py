"""
Top-level LangGraph state machine — the full research institute workflow.

This is the main entry point for running a research session. It wires
together the Director, Research Cycle, Debate Cycle, and Controller
into a looping state machine that runs until budget exhaustion,
convergence, or max iterations.

Flow:
  START → director_plan → spawn_squids
    → research_cycle → debate_cycle → controller_eval
      → (continue? loop back to research_cycle)
      → (stop? synthesize → END)
"""

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from langgraph.graph import StateGraph, START, END

from src.agents.controller import ControllerAgent
from src.agents.director import DirectorAgent
from src.config import Settings, settings as default_settings
from src.events.bus import EventBus
from src.graph.queries import GraphQueries
from src.graph.repository import GraphRepository
from src.llm.client import LLMClient
from src.llm.prompts import (
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_REPORT,
    ITERATION_BRIEFING_PROMPT,
    EXPERIMENT_INTERPRETATION_PROMPT,
)
from src.models.agent_state import AgentInfo, InstituteState, BeliefCluster
from src.models.archetype import Archetype, spawn_persona_from_archetype
from src.models.claim import Finding
from src.models.events import Event, EventType
from src.models.experiment import FindingInterpretation
from src.orchestration.debate_cycle import DebateCycleBuilder
from src.rag.indexer import RAGIndexer
from src.rag.retriever import RAGRetriever
from src.sandbox.runner import SandboxRunner
from src.search.arxiv import ArxivSearch
from src.search.tavily import TavilySearch
from src.workspace.manager import WorkspaceManager
from src.memory.hindsight_client import AgentMemory
from src.memory.server import HindsightManager

logger = logging.getLogger(__name__)


class InstituteGraphBuilder:
    """
    Builds the top-level LangGraph that orchestrates the entire institute.

    This graph manages the full research lifecycle:
    1. Director decomposes the question
    2. Squids are spawned for each subproblem
    3. Research and debate cycles alternate in a loop
    4. Controller decides when to stop
    5. Synthesizer produces the final report

    Usage:
        builder = InstituteGraphBuilder(llm, graph, queries, ...)
        compiled = builder.build()
        result = await compiled.ainvoke({
            "research_question": "What mechanisms drive X?",
            "budget_total": 500,
            "max_iterations": 5,
        })
    """

    def __init__(
        self,
        llm: LLMClient,
        graph: GraphRepository,
        queries: GraphQueries,
        retriever: RAGRetriever,
        indexer: RAGIndexer | None,
        event_bus: EventBus,
        sandbox: SandboxRunner,
        tavily: TavilySearch | None = None,
        arxiv_search: ArxivSearch | None = None,
        config: Settings | None = None,
        workspace_manager: WorkspaceManager | None = None,
        memory_manager: HindsightManager | None = None,
    ) -> None:
        self._llm = llm
        self._graph = graph
        self._queries = queries
        self._retriever = retriever
        self._indexer = indexer
        self._bus = event_bus
        self._sandbox = sandbox
        self._tavily = tavily
        self._arxiv = arxiv_search
        self._config = config or default_settings
        self._workspace_manager = workspace_manager
        self._memory_manager = memory_manager

        # Agent instances
        self._director = DirectorAgent(llm, event_bus, config=self._config)
        self._controller = ControllerAgent(
            llm, queries, event_bus, config=self._config
        )

        self._debate_builder = DebateCycleBuilder(
            llm, graph, queries, event_bus, config=self._config,
        )

    async def _update_budget_state(
        self,
        state: InstituteState,
    ) -> dict[str, Any]:
        """
        Update budget tracking fields from LLM usage data.

        This method:
        1. Adds token usage from the LLM client
        2. Adds dollar cost from the LLM client
        3. Checks for 90% threshold and sets budget_warning flag
        4. Emits BUDGET_WARNING event if crossing threshold

        Args:
            state: Current institute state

        Returns:
            Dict with updated budget fields to merge into state
        """
        tokens_used = self._llm.total_tokens
        dollars_used = self._llm.total_cost
        
        # Connect user's dollar budget
        budget_total_usd = float(state.get("budget_total_usd", self._config.default_budget_usd))

        budget_warning = state.get("budget_warning", False)
        # Use dollars for tracking usage_ratio
        usage_ratio = dollars_used / max(0.01, budget_total_usd)

        if usage_ratio >= 0.9 and not budget_warning:
            budget_warning = True
            await self._bus.publish(Event(
                event_type=EventType.BUDGET_WARNING,
                session_id=state.get("session_id", ""),
                payload={
                    "budget_total_usd": budget_total_usd,
                    "tokens_used": tokens_used,
                    "dollars_used": dollars_used,
                    "percentage": round(usage_ratio * 100, 1),
                },
            ))

        # Track remaining budget in dollars
        budget_remaining_usd = max(0.0, budget_total_usd - dollars_used)

        return {
            "budget_total_usd": budget_total_usd,
            "budget_remaining_usd": budget_remaining_usd,
            "tokens_used": tokens_used,
            "dollars_used": dollars_used,
            "budget_warning": budget_warning,
        }

    def build(self) -> Any:
        """
        Build and compile the full institute graph.

        Returns a compiled LangGraph that can be invoked with ainvoke().
        """
        graph = StateGraph(InstituteState)

        # Nodes
        graph.add_node("init_session", self._init_session)
        graph.add_node("director_plan", self._director_plan)
        graph.add_node("spawn_squids", self._spawn_squids)
        graph.add_node("research_cycle", self._research_cycle)
        graph.add_node("debate_cycle", self._debate_cycle)
        graph.add_node("controller_eval", self._controller_eval)
        graph.add_node("iteration_briefing", self._iteration_briefing)
        graph.add_node("increment_iteration", self._increment_iteration)
        graph.add_node("synthesize", self._synthesize)

        # Edges
        graph.add_edge(START, "init_session")
        graph.add_edge("init_session", "director_plan")
        graph.add_edge("director_plan", "spawn_squids")
        graph.add_edge("spawn_squids", "research_cycle")
        graph.add_edge("research_cycle", "debate_cycle")
        graph.add_edge("debate_cycle", "controller_eval")

        # Conditional: continue loop or stop
        graph.add_conditional_edges(
            "controller_eval",
            self._should_continue,
            {
                "continue": "iteration_briefing",
                "stop": "synthesize",
            },
        )
        graph.add_edge("iteration_briefing", "increment_iteration")
        graph.add_edge("increment_iteration", "research_cycle")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ── Node implementations ─────────────────────────────────────────

    async def _init_session(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """Initialize a new research session."""
        self._llm.reset_usage()
        session_id = state.get("session_id") or uuid4().hex[: self._config.session_id_length]

        # Connect user's dollar budget
        budget_total_usd = float(state.get("budget_total_usd", self._config.default_budget_usd))

        await self._bus.publish(Event(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            payload={"question": state.get("research_question", "")},
        ))

        await self._bus.publish(Event(
            event_type=EventType.STATE_SNAPSHOT,
            session_id=session_id,
            payload={
                "question": state.get("research_question", ""),
                "num_agents": state.get("num_agents", self._config.default_agents),
                "budget_total_usd": budget_total_usd,
                "budget_remaining_usd": budget_total_usd,
                "tokens_used": 0,
                "dollars_used": 0.0,
                "budget_warning": False,
                "max_iterations": state.get("max_iterations", self._config.default_iterations),
                "iteration": 0,
                "status": "active",
            },
        ))

        return {
            "session_id": session_id,
            "iteration": 0,
            "budget_total_usd": budget_total_usd,
            "budget_remaining_usd": budget_total_usd,
            "tokens_used": 0,
            "dollars_used": 0.0,
            "budget_warning": False,
            "should_stop": False,
            "coverage": {},
            "agents": [],
            "subproblems": [],
            "archetypes": [],
            "open_questions": [],
            "key_assumptions": [],
            "pending_experiments": [],
            "debate_queue": [],
            "belief_clusters": [],
            "last_recluster_iteration": -1,
            "artifacts_this_iteration": [],
            "controller_directives": [],
            "iteration_summary": "",
            "source_ids": state.get("source_ids", []),
            "events": [],
            "num_agents": state.get(
                "num_agents",
                self._default_num_agents(
                    state.get("budget_total_usd", self._config.default_budget_usd)
                ),
            ),
        }

    async def _director_plan(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """Run the Director to decompose the research question."""
        result = await self._director.run(state)
        await self._bus.publish(Event(
            event_type=EventType.STATE_SNAPSHOT,
            session_id=state.get("session_id", ""),
            payload={
                "subproblems": result.get("subproblems", []),
                "archetypes": result.get("archetypes", []),
                "open_questions": result.get("open_questions", []),
                "key_assumptions": result.get("key_assumptions", []),
            },
        ))
        budget_updates = await self._update_budget_state(state)
        return {
            **result,
            **budget_updates,
        }

    async def _spawn_squids(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Create squid agents from archetypes with persona diversity.

        Each agent gets a unique persona instantiated from a Director-
        designed archetype, with slight trait randomization. Multiple
        agents can share a subproblem if num_agents > len(subproblems).

        Budget is distributed across agents weighted by subproblem
        priority (higher priority = more budget).
        """
        subproblems = state.get("subproblems", [])
        archetype_dicts = state.get("archetypes", [])
        session_id = state.get("session_id", "")
        num_agents = state.get("num_agents", len(subproblems))
        total_budget_usd = state.get(
            "budget_remaining_usd", self._config.default_budget_usd
        )

        # Reconstruct Archetype models from serialized dicts
        archetypes = [Archetype(**d) for d in archetype_dicts] if archetype_dicts else []

        # Fallback: if no archetypes, create defaults
        if not archetypes:
            archetypes = self._default_archetypes()

        agents: list[AgentInfo] = []

        # Distribute budget by subproblem priority (lower number = higher priority)
        priority_sum = sum(1.0 / sp["priority"] for sp in subproblems) if subproblems else 1.0

        for i in range(num_agents):
            agent_id = f"{session_id}-squid-{i+1}"

            # Round-robin subproblem assignment
            sp = subproblems[i % len(subproblems)]

            # Round-robin archetype assignment
            archetype = archetypes[i % len(archetypes)]

            # Create persona from archetype with randomized traits
            persona = spawn_persona_from_archetype(
                archetype=archetype,
                agent_id=agent_id,
                session_id=session_id,
                specialty_override=None,
            )

            # Generate a name from archetype + index
            name = f"Dr. {archetype.name.split()[0]}-{i+1}"

            # Per-agent budget weighted by subproblem priority
            priority_weight = (1.0 / sp["priority"]) / priority_sum
            agent_budget_usd = max(
                self._config.min_agent_budget_usd,
                total_budget_usd * priority_weight / max(1, num_agents // len(subproblems)),
            )

            # Create agent workspace if workspace layer is active
            workspace_path = ""
            if self._workspace_manager:
                ws_root = await self._workspace_manager.create_workspace(
                    agent_id=agent_id,
                    session_id=session_id,
                    agent_name=name,
                    subproblem=sp["question"],
                )
                workspace_path = str(ws_root)

            agents.append(AgentInfo(
                agent_id=agent_id,
                name=name,
                line_of_inquiry=sp["question"],
                subproblem_id=sp["id"],
                status="active",
                persona=persona.model_dump(),
                budget_allocated_usd=agent_budget_usd,
                budget_used_usd=0.0,
                consecutive_empty_iterations=0,
                workspace_path=workspace_path,
            ))

            # Update subproblem with assigned agent
            if agent_id not in sp["assigned_agent"]:
                sp["assigned_agent"].append(agent_id)

            await self._bus.publish(Event(
                event_type=EventType.AGENT_SPAWNED,
                agent_id=agent_id,
                payload={
                    "name": name,
                    "inquiry": sp["question"],
                    "archetype": archetype.name,
                    "persona_id": persona.id,
                    "model_tier": persona.model_tier,
                },
            ))

        await self._bus.publish(Event(
            event_type=EventType.STATE_SNAPSHOT,
            session_id=session_id,
            payload={
                "agents": agents,
                "subproblems": subproblems,
            },
        ))

        return {
            "agents": agents,
            "subproblems": subproblems,
            "belief_clusters": [],
            "last_recluster_iteration": -1,
        }

    def _default_archetypes(self) -> list[Archetype]:
        """Minimum archetype set when Director doesn't produce any."""
        return [
            Archetype(**data) for data in self._config.fallback_archetypes
        ]

    def _default_num_agents(self, budget_total_usd: float) -> int:
        """Infer a reasonable agent count from budget when one is not supplied."""
        derived = int(budget_total_usd // self._config.budget_per_agent_target)
        return derived or self._config.default_agents

    async def _research_cycle(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Execute one research cycle — all squids work in parallel.

        Uses asyncio.gather for concurrent execution. At 100 agents,
        this means wall-clock time is limited by the slowest LLM call,
        not N × call_time. Agents with exhausted per-agent budgets
        are skipped.
        """
        import asyncio
        from src.agents.squid import SquidAgent
        from src.models.agent_state import SquidState

        # Build SquidState for each active agent
        tasks = []
        active_agents = []
        for agent in state.get("agents", []):
            if agent["status"] != "active":
                continue

            # Skip agents with exhausted per-agent budgets
            if agent.get("budget_used_usd", 0) >= agent.get(
                "budget_allocated_usd", float("inf")
            ):
                continue

            subproblem = None
            for sp in state.get("subproblems", []):
                if sp["id"] == agent["subproblem_id"]:
                    subproblem = sp
                    break

            if not subproblem:
                continue

            # Create per-squid workspace tools if workspace is available
            workspace_tools = None
            session_id = state.get("session_id", "")
            agent_id = agent["agent_id"]
            if self._workspace_manager and agent.get("workspace_path"):
                from src.workspace.submitter import ExperimentSubmitter
                from src.workspace.memory_enforcer import MemoryEnforcer
                from src.agents.workspace_tools import WorkspaceTools

                opencode_server = await self._workspace_manager.get_or_start_server(
                    agent_id, session_id
                )
                submitter = ExperimentSubmitter(self._graph, self._workspace_manager)
                enforcer = MemoryEnforcer(self._workspace_manager, self._config)
                workspace_tools = WorkspaceTools(
                    agent_id=agent_id,
                    session_id=session_id,
                    workspace_manager=self._workspace_manager,
                    opencode_server=opencode_server,
                    submitter=submitter,
                    enforcer=enforcer,
                    event_bus=self._bus,
                )

            # Create per-agent memory if Hindsight is available
            agent_memory = None
            if self._memory_manager and self._memory_manager.is_running:
                agent_memory = AgentMemory(
                    agent_id=agent["agent_id"],
                    session_id=session_id,
                    client=self._memory_manager.client,
                    config=self._config,
                )

            squid = SquidAgent(
                llm=self._llm,
                graph=self._graph,
                retriever=self._retriever,
                indexer=self._indexer,
                event_bus=self._bus,
                tavily=self._tavily,
                arxiv_search=self._arxiv,
                config=self._config,
                workspace_tools=workspace_tools,
                graph_queries=self._queries,
                agent_memory=agent_memory,
            )

            sci_state = SquidState(
                agent_id=agent["agent_id"],
                agent_name=agent["name"],
                subproblem=subproblem,
                session_id=session_id,
                iteration=state.get("iteration", 0),
                budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
                persona=agent.get("persona", {}),
                iteration_summary=state.get("iteration_summary", ""),
                workspace_path=agent.get("workspace_path", ""),
            )

            tasks.append(squid.run(sci_state))
            active_agents.append(agent)

        # Run all squids in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_artifacts: list[str] = []
        updated_agents = list(state.get("agents", []))

        for agent, result in zip(active_agents, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Agent {agent['agent_id'][:12]} ({agent.get('name', '?')}) "
                    f"crashed: {type(result).__name__}: {result}",
                    exc_info=result,
                )
                await self._bus.publish(Event(
                    event_type=EventType.ERROR,
                    agent_id=agent["agent_id"],
                    payload={
                        "error": f"Agent crashed: {type(result).__name__}: {result}",
                    },
                ))
                continue

            all_artifacts.extend(result.get("hypotheses_created", []))
            all_artifacts.extend(result.get("notes_created", []))

            # Update per-agent budget usage and empty iteration tracking
            for ua in updated_agents:
                if ua["agent_id"] == agent["agent_id"]:
                    # Track precise token cost used during this agent's run
                    ua["budget_used_usd"] = ua.get("budget_used_usd", 0.0) + result.get("spent_usd", 0.0)
                    produced = (
                        len(result.get("hypotheses_created", []))
                        + len(result.get("notes_created", []))
                    )
                    if produced == 0:
                        ua["consecutive_empty_iterations"] = (
                            ua.get("consecutive_empty_iterations", 0) + 1
                        )
                    else:
                        ua["consecutive_empty_iterations"] = 0
                    break

        # Run pending experiments in parallel
        pending = await self._graph.get_by_label(
            "Experiment",
            filters={
                "status": "pending",
                "session_id": state.get("session_id", ""),
            },
            limit=self._config.graph_pending_experiments_limit,
        )
        if pending:
            semaphore = asyncio.Semaphore(self._config.max_parallel_experiments)

            async def _run_capped(exp_data: dict) -> None:
                async with semaphore:
                    await self._execute_single_experiment(
                        exp_data, state.get("session_id", "")
                    )

            experiment_tasks = [_run_capped(exp) for exp in pending]
            await asyncio.gather(*experiment_tasks, return_exceptions=True)

        budget_updates = await self._update_budget_state(state)

        return {
            "agents": updated_agents,
            "artifacts_this_iteration": all_artifacts,
            **budget_updates,
            "pending_experiments": [],
        }

    async def _execute_single_experiment(
        self, exp_data: dict, session_id: str
    ) -> None:
        """Execute a single experiment, create result, and auto-generate Finding."""
        exp_id = exp_data["id"]
        input_data = self._decode_input_data(exp_data.get("spec_input_data", ""))
        await self._bus.publish(Event(
            event_type=EventType.EXPERIMENT_STARTED,
            artifact_id=exp_id,
            payload={
                "hypothesis_id": exp_data.get("hypothesis_id", ""),
                "expected_outcome": exp_data.get("spec_expected_outcome", ""),
                "code_preview": exp_data.get("spec_code", "")[:240],
                "input_data": input_data,
            },
        ))
        try:
            run_result = await self._sandbox.run_experiment(
                exp_id, exp_data
            )
            from src.models.experiment import ExperimentResult
            exp_result = ExperimentResult(
                experiment_id=exp_id,
                stdout=run_result.get("stdout", ""),
                stderr=run_result.get("stderr", ""),
                exit_code=run_result.get("exit_code", -1),
                artifacts=run_result.get("artifacts", {}),
                execution_time_seconds=run_result.get("execution_time", 0.0),
                created_by=exp_data.get("created_by", "system"),
            )
            await self._graph.create(exp_result)
            await self._graph.link_experiment_to_result(
                exp_id, exp_result.id
            )

            exit_code = int(run_result.get("exit_code", -1))
            if exit_code == 0:
                await self._graph.update(exp_id, {"status": "completed"})
                await self._bus.publish(Event(
                    event_type=EventType.EXPERIMENT_COMPLETED,
                    artifact_id=exp_id,
                    payload={
                        "result_id": exp_result.id,
                        "exit_code": exit_code,
                        "execution_time": run_result.get("execution_time", 0.0),
                        "stdout_preview": run_result.get("stdout", "")[:240],
                        "stderr_preview": run_result.get("stderr", "")[:240],
                        "artifacts": run_result.get("artifacts", {}),
                        "input_data": input_data,
                    },
                ))
            else:
                await self._graph.update(exp_id, {"status": "failed"})
                await self._bus.publish(Event(
                    event_type=EventType.EXPERIMENT_FAILED,
                    artifact_id=exp_id,
                    payload={
                        "exit_code": exit_code,
                        "execution_time": run_result.get("execution_time", 0.0),
                        "error": (
                            run_result.get("stderr", "")
                            or run_result.get("stdout", "")
                            or "Experiment exited with a non-zero status."
                        )[:400],
                        "expected_outcome": exp_data.get("spec_expected_outcome", ""),
                        "code_preview": exp_data.get("spec_code", "")[:240],
                        "input_data": input_data,
                        "stdout_preview": run_result.get("stdout", "")[:240],
                    },
                ))

            # Auto-generate Finding from ExperimentResult
            finding = await self._interpret_experiment_result(
                exp_data, exp_result, session_id
            )
            if finding:
                await self._graph.create(finding)
                if finding.hypothesis_id:
                    await self._graph.link_finding_to_hypothesis(
                        finding.id, finding.hypothesis_id
                    )
                # Embed the finding for RAG retrieval
                if self._indexer:
                    await self._indexer.index_artifact(finding)
                logger.info(
                    "Auto-generated Finding %s (%s, conf=%.2f) for experiment %s",
                    finding.id, finding.conclusion_type, finding.confidence, exp_id,
                )

        except Exception as exc:
            await self._graph.update(exp_id, {"status": "failed"})
            await self._bus.publish(Event(
                event_type=EventType.EXPERIMENT_FAILED,
                artifact_id=exp_id,
                payload={
                    "error": str(exc),
                    "expected_outcome": exp_data.get("spec_expected_outcome", ""),
                    "code_preview": exp_data.get("spec_code", "")[:240],
                    "input_data": input_data,
                },
            ))

    async def _interpret_experiment_result(
        self,
        exp_data: dict,
        result: Any,
        session_id: str,
    ) -> Finding | None:
        """
        Ask LLM to interpret experiment output into a structured Finding.

        Uses the 'fast' model tier — this is a classification task, not reasoning.
        The LLM sees: hypothesis text, expected outcome, actual stdout/stderr, exit code.
        It produces: conclusion_type (supports/refutes/inconclusive/partial),
                     confidence (0-1), explanation text.
        """
        try:
            # Fetch hypothesis text for context
            hyp_id = exp_data.get("hypothesis_id", "")
            hyp_node = await self._graph.get(hyp_id) if hyp_id else None
            hypothesis_text = (
                hyp_node.get("text", "") if hyp_node else "Unknown hypothesis"
            )

            prompt = EXPERIMENT_INTERPRETATION_PROMPT.format(
                hypothesis_text=hypothesis_text,
                expected_outcome=exp_data.get("spec_expected_outcome", ""),
                actual_stdout=result.stdout[:2000],
                actual_stderr=result.stderr[:500],
                exit_code=result.exit_code,
            )
            interpretation = await self._llm.complete_structured(
                prompt=prompt,
                response_model=FindingInterpretation,
                system="You are a scientific experiment interpreter. Be objective.",
                temperature=self._config.temperature_default_structured,
            )
            return Finding(
                text=interpretation.text,
                hypothesis_id=hyp_id,
                experiment_id=exp_data["id"],
                conclusion_type=interpretation.conclusion_type,
                confidence=interpretation.confidence,
                created_by="system-interpreter",
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to interpret experiment %s: %s", exp_data["id"], exc
            )
            return None

    @staticmethod
    def _decode_input_data(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    async def _debate_cycle(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Execute one debate cycle using cluster-based routing.

        Delegates to the DebateCycleBuilder subgraph which handles:
        1. Belief cluster computation (every 2-3 iterations)
        2. Intra-cluster review (agents refine shared positions)
        3. Inter-cluster debate (opposing clusters challenge each other)
        4. Counter-responses (challenged authors rebut)
        5. Adjudication (provisional rulings on contested hypotheses)
        6. Contradiction resolution

        After the debate, confidence propagation runs to update hypothesis
        confidence from accumulated Findings.

        At 100 agents this produces ~350 review calls instead of ~9,900.
        """
        cost_before = self._llm.total_cost

        # Build and run the debate subgraph
        debate_graph = self._debate_builder.build()
        compiled = debate_graph.compile()
        result = await compiled.ainvoke(state)

        # Close the evidence loop — propagate findings to confidence
        session_id = state.get("session_id", "")
        try:
            from src.orchestration.evidence import propagate_confidence
            confidence_updates = await propagate_confidence(
                self._graph, self._queries, session_id,
                self._bus, self._config,
            )
            if confidence_updates:
                logger.info(
                    "Confidence propagation updated %d hypotheses: %s",
                    len(confidence_updates),
                    {k: f"{v:.3f}" for k, v in confidence_updates.items()},
                )
        except Exception as exc:
            logger.warning("Confidence propagation failed: %s", exc)

        # Extract relevant state updates from debate result
        return {
            "debate_queue": result.get("debate_queue", []),
            "belief_clusters": result.get("belief_clusters", state.get("belief_clusters", [])),
            "last_recluster_iteration": result.get(
                "last_recluster_iteration",
                state.get("last_recluster_iteration", -1),
            ),
            "pending_experiments": result.get("pending_experiments", []),
            "budget_remaining_usd": max(
                0.0,
                state.get("budget_remaining_usd", 0.0)
                - (self._llm.total_cost - cost_before)
            ),
        }

    async def _controller_eval(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """Run the Controller to evaluate progress."""
        result = await self._controller.evaluate(state)
        budget_updates = await self._update_budget_state(state)
        await self._bus.publish(Event(
            event_type=EventType.STATE_SNAPSHOT,
            session_id=state.get("session_id", ""),
            payload={
                "coverage": result.get("coverage", {}),
                "controller_directives": result.get("controller_directives", []),
                "should_stop": result.get("should_stop", False),
                "iteration": state.get("iteration", 0),
                "budget_remaining_usd": budget_updates["budget_remaining_usd"],
                "tokens_used": budget_updates["tokens_used"],
                "dollars_used": budget_updates["dollars_used"],
                "budget_warning": budget_updates["budget_warning"],
                "belief_clusters": state.get("belief_clusters", []),
                "pending_experiments": state.get("pending_experiments", []),
                "agents": result.get("agents", state.get("agents", [])),
            },
        ))
        return {
            **result,
            **budget_updates,
        }

    def _should_continue(self, state: InstituteState) -> str:
        """Decide whether to continue researching or stop."""
        if state.get("should_stop", False):
            return "stop"
        return "continue"

    async def _iteration_briefing(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Generate an institute-wide briefing for all agents.

        Summarizes: current best hypotheses, top contradictions,
        cluster positions, recommended focus areas, and convergence metrics.
        This briefing is injected into every squid's prompt in the next
        iteration so they know the institutional context.

        Hypotheses are sorted by evidence coverage (least-evidenced first)
        so the briefing naturally directs attention to gaps.
        """
        # Gather top hypotheses
        session_id = state.get("session_id", "")
        hypotheses = await self._queries.get_all_hypotheses(
            status="active",
            session_id=session_id,
        )

        # Sort by evidence coverage — least-evidenced first
        hyp_with_coverage = []
        for h in hypotheses:
            ctx = await self._queries.get_hypothesis_context(h["id"])
            finding_count = len(ctx.get("findings", []))
            exp_count = len(ctx.get("experiments", []))
            hyp_with_coverage.append((h, finding_count, exp_count))

        # Sort: least-evidenced hypotheses first (coverage gaps bubble up)
        hyp_with_coverage.sort(key=lambda x: (x[1], x[2]))

        top_hyp = "\n".join(
            f"- (confidence: {h.get('confidence', '?')}, "
            f"by: {h.get('created_by', '?')}, "
            f"findings: {fc}, experiments: {ec}): {h.get('text', '')[:150]}"
            for h, fc, ec in hyp_with_coverage[: self._config.briefing_top_hypotheses_limit]
        ) or "None yet."

        # Contradictions
        contradictions = await self._queries.get_session_contradictions(session_id=session_id)
        contra_text = "\n".join(
            f"- {c.get('source_text', '')[:80]} vs {c.get('target_text', '')[:80]}"
            for c in contradictions[: self._config.briefing_contradictions_limit]
        ) or "None."

        # Cluster summary
        clusters = state.get("belief_clusters", [])
        cluster_text = "\n".join(
            f"- Cluster {c['cluster_id']}: {len(c['agent_ids'])} agents, "
            f"{len(c.get('shared_hypotheses', []))} shared hypotheses, "
            f"{len(c.get('contested_hypotheses', []))} contested"
            for c in clusters
        ) or "No clusters formed yet."

        # Agent performance (from reputation tracker)
        from src.agents.reputation import ReputationTracker
        tracker = ReputationTracker(self._queries, config=self._config)
        agent_ids = [
            a["agent_id"] for a in state.get("agents", [])
            if a["status"] == "active"
        ]
        empty_map = {
            a["agent_id"]: a.get("consecutive_empty_iterations", 0)
            for a in state.get("agents", [])
        }
        all_metrics = await tracker.get_all_metrics(agent_ids, empty_map)
        perf_text = "\n".join(
            f"- {m.agent_id}: score={m.composite_score:.2f}, "
            f"hyp_active={m.hypotheses_active}, "
            f"refuted={m.hypotheses_refuted}, "
            f"empty_runs={m.consecutive_empty}"
            for m in all_metrics
        ) or "No performance data yet."

        # Include convergence metrics in briefing
        convergence_metrics = state.get("convergence_metrics", {})
        convergence_text = ""
        if convergence_metrics:
            convergence_text = (
                f"\n\n=== CONVERGENCE ===\n"
                f"Score: {convergence_metrics.get('convergence_score', 0):.2f} "
                f"(threshold: {self._config.convergence_threshold})\n"
                f"High-confidence: {convergence_metrics.get('high_confidence_count', 0)}"
                f"/{convergence_metrics.get('total_hypotheses', 0)}\n"
                f"Evidenced: {convergence_metrics.get('evidence_covered_count', 0)}"
                f"/{convergence_metrics.get('total_hypotheses', 0)}\n"
                f"Contradicted: {convergence_metrics.get('contradicted_count', 0)}"
            )

        prompt = ITERATION_BRIEFING_PROMPT.format(
            research_question=state.get("research_question", ""),
            iteration=state.get("iteration", 0),
            max_iterations=state.get(
                "max_iterations", self._config.default_iterations
            ),
            top_hypotheses=top_hyp,
            contradictions=contra_text,
            cluster_summary=cluster_text + convergence_text,
            agent_performance=perf_text,
        )

        briefing = await self._llm.complete(
            prompt=prompt,
            system="You are the institute briefing system. Produce a clear, "
            "actionable briefing for all research agents.",
            temperature=self._config.temperature_briefing,
            max_tokens=self._config.max_tokens_briefing,
        )

        budget_updates = await self._update_budget_state(state)

        return {
            "iteration_summary": briefing,
            **budget_updates,
        }

    async def _increment_iteration(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """Increment the iteration counter."""
        return {
            "iteration": state.get("iteration", 0) + 1,
            "artifacts_this_iteration": [],
        }

    async def _synthesize(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Produce the final research synthesis report.

        Gathers all hypotheses, findings, experiment results, and
        agent contributions, then asks the LLM to produce a
        comprehensive research report.
        """
        # Gather all artifacts for synthesis
        session_id = state.get("session_id", "")
        hypotheses = await self._queries.get_all_hypotheses(status="active", session_id=session_id)
        all_hypotheses = await self._queries.get_all_hypotheses(status="active", session_id=session_id)
        refuted = await self._queries.get_all_hypotheses(status="refuted", session_id=session_id)
        all_hypotheses.extend(refuted)

        contradictions = await self._queries.get_session_contradictions(session_id=session_id)

        # Format for the synthesis prompt
        hyp_text = "\n".join(
            f"- [{h.get('status', 'active')}] (confidence: {h.get('confidence', '?')}, "
            f"by: {h.get('created_by', '?')}): {h.get('text', '')}"
            for h in all_hypotheses
        )

        findings_data = await self._graph.get_by_label(
            "Finding",
            filters={"session_id": session_id},
            limit=self._config.graph_findings_synthesis_limit,
        )
        findings_text = "\n".join(
            f"- ({f.get('conclusion_type', '?')}): {f.get('text', '')}"
            for f in findings_data
        )

        results_data = await self._graph.get_by_label(
            "ExperimentResult",
            filters={"session_id": session_id},
            limit=self._config.graph_experiment_results_limit,
        )
        results_text = "\n".join(
            f"- Exit {r.get('exit_code', '?')}: {r.get('stdout', '')[:200]}"
            for r in results_data
        )

        contradictions_text = "\n".join(
            f"- {c.get('source_text', '')[:100]} vs {c.get('target_text', '')[:100]}"
            for c in contradictions[: self._config.graph_contradictions_prompt_limit]
        )

        agent_text = "\n".join(
            f"- {a['name']} ({a['agent_id']}): {a['line_of_inquiry']}"
            for a in state.get("agents", [])
        )

        prompt = SYNTHESIZER_REPORT.format(
            research_question=state.get("research_question", ""),
            hypotheses=hyp_text or "None",
            findings=findings_text or "None",
            experiment_results=results_text or "None",
            contradictions=contradictions_text or "None",
            agent_contributions=agent_text or "None",
        )

        report = await self._llm.complete(
            prompt=prompt,
            system=SYNTHESIZER_SYSTEM,
            temperature=self._config.temperature_synthesizer,
            max_tokens=self._config.max_tokens_synthesizer,
        )

        # Snapshot and stop workspace servers before completing
        if self._workspace_manager:
            await self._workspace_manager.snapshot_session(session_id)
            await self._workspace_manager.stop_all_servers(session_id)

        await self._bus.publish(Event(
            event_type=EventType.RESEARCH_COMPLETED,
            session_id=session_id,
            payload={
                "iterations": state.get("iteration", 0),
                "budget_used": state.get("budget_total_usd", 0.0) - state.get("budget_remaining_usd", 0.0),
                "report_length": len(report),
            },
        ))

        return {
            "final_report": report,
            "events": [{
                "type": "final_report",
                "content": report,
            }],
        }
