"""
Institute Controller agent — manages research progress, budget, and agent lifecycle.

The Controller evaluates the overall state of research after each
iteration cycle and decides whether to continue, where to focus,
when to stop, and which agents to pause or reallocate budget from.

Uses reputation metrics to identify underperforming agents and
reallocate their remaining budget to productive ones.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.reputation import ReputationTracker
from src.config import Settings, settings as default_settings
from src.events.bus import EventBus
from src.graph.queries import GraphQueries
from src.llm.client import LLMClient
from src.llm.prompts import CONTROLLER_SYSTEM, CONTROLLER_EVALUATE
from src.models.agent_state import InstituteState
from src.models.events import Event, EventType


class ControllerOutput(BaseModel):
    """Structured output from the Controller's evaluation."""

    should_stop: bool = False
    reasoning: str = ""
    coverage_assessment: dict[str, float] = Field(default_factory=dict)
    directives: list[str] = Field(default_factory=list)
    priority_shifts: list[dict[str, Any]] = Field(default_factory=list)
    agents_to_pause: list[str] = Field(default_factory=list)


class ControllerAgent:
    """
    Evaluates research progress and decides next steps.

    The Controller runs after each research+debate cycle and determines:
    - Whether the research should stop (budget exhausted, convergence, etc.)
    - Coverage gaps (which subproblems need more attention)
    - Priority shifts (reorder subproblems based on findings)
    - Directives (specific instructions for the next cycle)

    Usage:
        controller = ControllerAgent(llm, queries, event_bus)
        state_update = await controller.evaluate(state)
    """

    def __init__(
        self,
        llm: LLMClient,
        queries: GraphQueries,
        event_bus: EventBus,
        config: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._queries = queries
        self._bus = event_bus
        self._config = config or default_settings
        self._reputation = ReputationTracker(queries, config=self._config)

    async def evaluate(self, state: InstituteState) -> dict[str, Any]:
        """
        Evaluate research progress, manage agent lifecycle, and produce directives.

        In addition to the original coverage/budget/stop evaluation,
        this now:
        1. Computes per-agent reputation metrics
        2. Pauses underperforming agents (2+ consecutive empty iterations)
        3. Reallocates paused agents' remaining budget to productive agents

        Args:
            state: Current institute state.

        Returns:
            State update dict with should_stop, coverage, directives,
            and updated agent list (with paused agents marked).
        """
        session_id = state.get("session_id", "")
        # Gather graph statistics
        stats = await self._queries.get_coverage_stats(session_id=session_id)
        contradictions = await self._queries.get_session_contradictions(
            session_id=session_id
        )
        agent_lookup = {
            agent["agent_id"]: agent.get("name", agent["agent_id"])
            for agent in state.get("agents", [])
        }

        # Compute per-agent reputation
        agents = list(state.get("agents", []))
        active_agents = [a for a in agents if a["status"] == "active"]
        empty_map = {
            a["agent_id"]: a.get("consecutive_empty_iterations", 0)
            for a in agents
        }
        all_metrics = await self._reputation.get_all_metrics(
            [a["agent_id"] for a in active_agents], empty_map
        )

        # Format state for the prompt
        subproblems_text = "\n".join(
            f"- [{sp['id']}] (priority {sp['priority']}): {sp['question']}"
            for sp in state.get("subproblems", [])
        )

        coverage = state.get("coverage", {})
        coverage_text = "\n".join(
            f"- {k}: {v:.0%}" for k, v in coverage.items()
        ) or "No coverage data yet."

        contradictions_text = "\n".join(
            f"- {c.get('source_text', '')[:100]} vs {c.get('target_text', '')[:100]}"
            for c in contradictions[: self._config.graph_contradictions_prompt_limit]
        ) or "None"

        stats_text = "\n".join(
            f"- {label}: {status_counts}"
            for label, status_counts in stats.items()
        )

        # Add agent performance to the prompt
        agent_perf_text = "\n".join(
            f"- {agent_lookup.get(m.agent_id, m.agent_id)} ({m.agent_id}): score={m.composite_score:.2f}, "
            f"hyp={m.hypotheses_active}active/{m.hypotheses_refuted}refuted, "
            f"empty_runs={m.consecutive_empty}"
            for m in all_metrics
        )

        # Compute convergence metrics
        from src.orchestration.convergence import compute_convergence
        convergence = await compute_convergence(
            self._queries, session_id, self._config
        )
        convergence_text = convergence.summary_text()

        prompt = CONTROLLER_EVALUATE.format(
            research_question=state.get("research_question", ""),
            subproblems=subproblems_text,
            coverage=coverage_text,
            iteration=state.get("iteration", 0),
            max_iterations=state.get(
                "max_iterations", self._config.default_iterations
            ),
            budget_remaining=state.get("budget_remaining_usd", 0.0),
            budget_total=state.get("budget_total_usd", self._config.default_budget_usd),
            graph_stats=(
                stats_text
                + f"\n\n=== AGENT PERFORMANCE ===\n{agent_perf_text}"
                + f"\n\n=== CONVERGENCE METRICS ===\n{convergence_text}"
            ),
            contradictions=contradictions_text,
        )

        output = await self._llm.complete_structured(
            prompt=prompt,
            response_model=ControllerOutput,
            system=CONTROLLER_SYSTEM,
            temperature=self._config.temperature_controller,
        )

        # Force stop if budget or iterations exhausted
        budget_usd = state.get("budget_remaining_usd", 0.0)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", self._config.default_iterations)

        # Convergence-based auto-stop
        converged = convergence.convergence_score >= self._config.convergence_threshold
        force_stop = budget_usd <= 0 or iteration >= max_iter
        should_stop = output.should_stop or force_stop or converged

        if force_stop and not output.should_stop:
            output.reasoning += (
                f" [FORCED STOP: budget=${budget_usd:.2f}, "
                f"iteration={iteration}/{max_iter}]"
            )
        elif converged and not output.should_stop:
            output.reasoning += (
                f" [CONVERGENCE STOP: score={convergence.convergence_score:.2f} "
                f">= threshold={self._config.convergence_threshold}]"
            )

        output.reasoning = self._humanize_agent_references(
            output.reasoning,
            agent_lookup,
        )
        output.directives = [
            self._humanize_agent_references(directive, agent_lookup)
            for directive in output.directives
        ]

        # Pause underperforming agents and reallocate budget
        paused_budget = 0
        for metrics in all_metrics:
            if self._reputation.should_pause(
                metrics,
                threshold=self._config.agent_pause_empty_threshold,
            ):
                for a in agents:
                    if a["agent_id"] == metrics.agent_id and a["status"] == "active":
                        a["status"] = "paused"
                        remaining = a.get("budget_allocated_usd", 0) - a.get("budget_used_usd", 0)
                        paused_budget += max(0, remaining)

                        await self._bus.publish(Event(
                            event_type=EventType.AGENT_ACTION,
                            agent_id=metrics.agent_id,
                            payload={
                                "action": "paused",
                                "reason": f"score={metrics.composite_score:.2f}, "
                                f"empty_runs={metrics.consecutive_empty}",
                                "budget_returned": remaining,
                            },
                        ))

        # Reallocate paused budget to top-performing active agents
        if paused_budget > 0:
            still_active = [a for a in agents if a["status"] == "active"]
            if still_active:
                per_agent_bonus = paused_budget // len(still_active)
                for a in still_active:
                    a["budget_allocated"] = a.get("budget_allocated", 0) + per_agent_bonus

        await self._bus.publish(Event(
            event_type=EventType.ITERATION_COMPLETED,
            agent_id="controller",
            payload={
                "iteration": iteration,
                "should_stop": should_stop,
                "reasoning": output.reasoning,
                "directives": output.directives,
                "agents_paused": [
                    m.agent_id for m in all_metrics
                    if self._reputation.should_pause(
                        m,
                        threshold=self._config.agent_pause_empty_threshold,
                    )
                ],
                "budget_reallocated": paused_budget,
            },
        ))

        return {
            "should_stop": should_stop,
            "coverage": output.coverage_assessment,
            "controller_directives": output.directives,
            "agents": agents,
            "convergence_metrics": convergence.to_dict(),
        }

    @staticmethod
    def _humanize_agent_references(
        text: str,
        agent_lookup: dict[str, str],
    ) -> str:
        """Replace backend ids and squid-N shorthand with display names."""
        if not text:
            return text

        rewritten = text
        for agent_id, agent_name in sorted(
            agent_lookup.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            rewritten = re.sub(
                rf"(?<!\\w){re.escape(agent_id)}(?!\\w)",
                agent_name,
                rewritten,
            )
            shorthand_match = re.search(r"(squid-\d+)$", agent_id)
            if shorthand_match:
                shorthand = shorthand_match.group(1)
                rewritten = re.sub(
                    rf"(?<!\\w){re.escape(shorthand)}(?!\\w)",
                    agent_name,
                    rewritten,
                )

        return rewritten
