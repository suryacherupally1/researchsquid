"""
Debate cycle subgraph — cluster-based challenge, counter, and adjudication.

This is where the real intellectual progress happens. Instead of O(N²)
every-agent-reviews-everything, debate is routed through belief clusters:

1. Compute/refresh belief clusters (every 2-3 iterations)
2. Intra-cluster: agents within a cluster refine shared positions
3. Inter-cluster: opposing clusters challenge each other's key hypotheses
4. Counter-responses: challenged authors get one rebuttal
5. Adjudication: contested hypotheses get a provisional ruling
6. Resolve: contradictions flagged for next cycle

At 100 agents in ~10 clusters, this produces ~350 review calls instead
of the 9,900 that full pairwise review would require.
"""

import asyncio
from typing import Any

from langgraph.graph import StateGraph, START, END

from src.agents.clustering import BeliefClusterer
from src.agents.reviewer import ReviewerAgent
from src.config import Settings, settings as default_settings
from src.events.bus import EventBus
from src.graph.queries import GraphQueries
from src.graph.repository import GraphRepository
from src.llm.client import LLMClient
from src.llm.prompts import (
    REVIEWER_SYSTEM,
    SQUID_COUNTER_RESPONSE,
    ADJUDICATOR_PROMPT,
)
from src.models.agent_state import BeliefCluster, InstituteState
from src.models.events import Event, EventType


class DebateCycleBuilder:
    """
    Builds the LangGraph subgraph for one debate iteration.

    The debate cycle uses belief-based clustering to route reviews
    efficiently. Only agents who actually disagree end up debating
    each other.

    Usage:
        builder = DebateCycleBuilder(llm, graph, queries, event_bus)
        graph = builder.build()
    """

    def __init__(
        self,
        llm: LLMClient,
        graph: GraphRepository,
        queries: GraphQueries,
        event_bus: EventBus,
        config: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._graph = graph
        self._queries = queries
        self._bus = event_bus
        self._config = config or default_settings

        self._reviewer = ReviewerAgent(
            llm=llm,
            graph=graph,
            queries=queries,
            event_bus=event_bus,
            config=self._config,
        )
        self._clusterer = BeliefClusterer(queries, config=self._config)

    def build(self) -> StateGraph:
        """Build and return the debate cycle subgraph."""

        graph = StateGraph(InstituteState)

        graph.add_node("start_debate", self._start_debate)
        graph.add_node("compute_clusters", self._compute_clusters)
        graph.add_node("intra_cluster_review", self._intra_cluster_review)
        graph.add_node("inter_cluster_debate", self._inter_cluster_debate)
        graph.add_node("counter_responses", self._counter_responses)
        graph.add_node("adjudicate", self._adjudicate)
        graph.add_node("resolve_contradictions", self._resolve_contradictions)

        graph.add_edge(START, "start_debate")
        graph.add_edge("start_debate", "compute_clusters")
        graph.add_edge("compute_clusters", "intra_cluster_review")
        graph.add_edge("intra_cluster_review", "inter_cluster_debate")
        graph.add_edge("inter_cluster_debate", "counter_responses")
        graph.add_edge("counter_responses", "adjudicate")
        graph.add_edge("adjudicate", "resolve_contradictions")
        graph.add_edge("resolve_contradictions", END)

        return graph

    async def _start_debate(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """Signal the start of a debate cycle."""
        await self._bus.publish(Event(
            event_type=EventType.DEBATE_STARTED,
            payload={"iteration": state.get("iteration", 0)},
        ))
        return {}

    async def _compute_clusters(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Recompute belief clusters if due (hybrid: every 2-3 iterations).

        On the first iteration, clusters are formed from archetype
        similarity. On subsequent re-cluster iterations, they're
        based on actual belief vectors (who supports/contradicts what).
        """
        iteration = state.get("iteration", 0)
        last_recluster = state.get("last_recluster_iteration", -1)
        existing_clusters = state.get("belief_clusters", [])

        # Re-cluster at configured interval, or if no clusters exist yet
        should_recluster = (
            not existing_clusters
            or (iteration - last_recluster) >= self._config.recluster_interval
        )

        if not should_recluster:
            return {}

        agent_ids = [
            a["agent_id"]
            for a in state.get("agents", [])
            if a["status"] == "active"
        ]
        session_id = state.get("session_id", "")

        if len(agent_ids) < 2:
            return {}

        clusters = await self._clusterer.cluster_agents(
            agent_ids,
            session_id=session_id,
        )

        agent_lookup = {
            agent["agent_id"]: agent.get("name", agent["agent_id"])
            for agent in state.get("agents", [])
        }
        cluster_summaries = [
            {
                "cluster_id": cluster["cluster_id"],
                "members": [
                    {
                        "agent_id": agent_id,
                        "name": agent_lookup.get(agent_id, agent_id),
                    }
                    for agent_id in cluster["agent_ids"]
                ],
                "shared_hypotheses": cluster.get("shared_hypotheses", []),
                "contested_hypotheses": cluster.get("contested_hypotheses", []),
            }
            for cluster in clusters
        ]

        await self._bus.publish(Event(
            event_type=EventType.CLUSTERS_COMPUTED,
            agent_id="debate-system",
            payload={
                "num_clusters": len(clusters),
                "cluster_sizes": [len(c["agent_ids"]) for c in clusters],
                "clusters": cluster_summaries,
            },
        ))

        return {
            "belief_clusters": clusters,
            "last_recluster_iteration": iteration,
        }

    async def _intra_cluster_review(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Within each cluster, agents review each other's work briefly.

        This is lighter than inter-cluster debate — agents in the same
        cluster broadly agree, so reviews focus on refinement and
        strengthening shared positions rather than challenges.

        Runs clusters in parallel for speed.
        """
        clusters = state.get("belief_clusters", [])
        if not clusters:
            # Fallback: no clusters, do legacy all-vs-all (capped)
            return await self._legacy_review(state)

        review_jobs: list[dict[str, Any]] = []
        review_plan_preview: list[dict[str, Any]] = []
        for cluster in clusters:
            members = cluster["agent_ids"]
            if len(members) <= 1:
                continue
            for reviewer_id in members:
                peers = [aid for aid in members if aid != reviewer_id][
                    : self._config.max_intra_cluster_peers
                ]
                for peer_id in peers:
                    peer_hypotheses = await self._queries.get_agent_hypotheses(
                        peer_id,
                        session_id=state.get("session_id", ""),
                    )
                    for hypothesis in peer_hypotheses[: self._config.max_hypotheses_per_peer]:
                        job = {
                            "cluster_id": cluster["cluster_id"],
                            "reviewer_id": reviewer_id,
                            "peer_id": peer_id,
                            "hypothesis_id": hypothesis["id"],
                            "hypothesis_text": hypothesis.get("text", ""),
                        }
                        review_jobs.append(job)
                        if len(review_plan_preview) < 8:
                            review_plan_preview.append(job)

        planned_reviews = len(review_jobs)

        await self._bus.publish(Event(
            event_type=EventType.INTRA_CLUSTER_REVIEW_STARTED,
            agent_id="debate-system",
            payload={
                "clusters": len(clusters),
                "planned_reviews": planned_reviews,
                "review_plan_preview": review_plan_preview,
            },
        ))

        if not review_jobs:
            await self._bus.publish(Event(
                event_type=EventType.INTRA_CLUSTER_REVIEW_COMPLETED,
                agent_id="debate-system",
                payload={
                    "completed_reviews": 0,
                    "experiments_proposed": 0,
                },
            ))
            return {}

        new_experiments = []
        completed_reviews = 0
        failed_reviews = 0
        total_reviews = len(review_jobs)

        async def run_review(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | Exception]:
            try:
                result = await self._reviewer.review_hypothesis(
                    job["hypothesis_id"],
                    job["reviewer_id"],
                )
                return job, result
            except Exception as exc:  # pragma: no cover - defensive logging
                return job, exc

        for task in asyncio.as_completed([run_review(job) for job in review_jobs]):
            job, result = await task
            if isinstance(result, Exception):
                failed_reviews += 1
            elif result:
                completed_reviews += 1
                new_experiments.extend(result.get("experiments_proposed", []))
            else:
                failed_reviews += 1

            await self._bus.publish(Event(
                event_type=EventType.INTRA_CLUSTER_REVIEW_PROGRESS,
                agent_id="debate-system",
                payload={
                    "completed_reviews": completed_reviews,
                    "failed_reviews": failed_reviews,
                    "total_reviews": total_reviews,
                    "reviewer_id": job["reviewer_id"],
                    "peer_id": job["peer_id"],
                    "hypothesis_id": job["hypothesis_id"],
                    "hypothesis_text": job.get("hypothesis_text", ""),
                },
            ))

        await self._bus.publish(Event(
            event_type=EventType.INTRA_CLUSTER_REVIEW_COMPLETED,
            agent_id="debate-system",
            payload={
                "completed_reviews": completed_reviews,
                "failed_reviews": failed_reviews,
                "experiments_proposed": len(new_experiments),
            },
        ))

        return {"pending_experiments": new_experiments}

    async def _inter_cluster_debate(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Pair opposing clusters and have representatives challenge
        each other's key hypotheses.

        The cluster representative (first agent) challenges hypotheses
        that the opposing cluster supports but their cluster doesn't.
        This is where genuine disagreements surface.
        """
        clusters = state.get("belief_clusters", [])
        if len(clusters) < 2:
            return {}

        debate_pairs = await self._clusterer.form_debate_pairs(clusters)
        enriched_pairs = []
        for pair in debate_pairs:
            hypothesis = await self._graph.get(pair["target_hypothesis_id"])
            enriched_pairs.append({
                **pair,
                "target_hypothesis_text": (hypothesis or {}).get("text", ""),
                "target_owner_id": (hypothesis or {}).get("created_by", ""),
            })

        await self._bus.publish(Event(
            event_type=EventType.INTER_CLUSTER_DEBATE_STARTED,
            agent_id="debate-system",
            payload={
                "pairs": len(enriched_pairs),
                "pair_preview": enriched_pairs[:6],
            },
        ))

        new_experiments = []
        completed_pairs = 0
        failed_pairs = 0
        total_pairs = len(enriched_pairs)

        async def run_debate(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | Exception]:
            try:
                result = await self._reviewer.review_hypothesis(
                    pair["target_hypothesis_id"],
                    pair["challenger_id"],
                )
                return pair, result
            except Exception as exc:  # pragma: no cover - defensive logging
                return pair, exc

        for task in asyncio.as_completed([run_debate(pair) for pair in enriched_pairs]):
            pair, result = await task
            if isinstance(result, Exception):
                failed_pairs += 1
            elif result:
                completed_pairs += 1
                new_experiments.extend(result.get("experiments_proposed", []))
            else:
                failed_pairs += 1

            await self._bus.publish(Event(
                event_type=EventType.INTER_CLUSTER_DEBATE_PROGRESS,
                agent_id="debate-system",
                payload={
                    "completed_pairs": completed_pairs,
                    "failed_pairs": failed_pairs,
                    "total_pairs": total_pairs,
                    "challenger_id": pair["challenger_id"],
                    "target_owner_id": pair.get("target_owner_id", ""),
                    "target_hypothesis_id": pair["target_hypothesis_id"],
                    "target_hypothesis_text": pair.get("target_hypothesis_text", ""),
                },
            ))

        await self._bus.publish(Event(
            event_type=EventType.INTER_CLUSTER_DEBATE_COMPLETED,
            agent_id="debate-system",
            payload={
                "completed_pairs": completed_pairs,
                "failed_pairs": failed_pairs,
                "experiments_proposed": len(new_experiments),
            },
        ))

        return {
            "pending_experiments": (
                state.get("pending_experiments", []) + new_experiments
            ),
        }

    async def _counter_responses(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Give challenged hypothesis authors one rebuttal opportunity.

        For each hypothesis that received a "challenge" or "refute"
        verdict in this debate round, the original author can:
        - Accept the critique (mark for revision)
        - Rebut with counter-evidence
        - Request an experiment to settle the dispute
        """
        # Find hypotheses challenged in this round (have recent
        # "challenge" or "refute" findings)
        from src.models.claim import Finding
        from src.models.message import Message, MessageType

        # Get recent findings from this iteration with challenge/refute conclusions
        all_findings = await self._graph.get_by_label(
            "Finding",
            filters={"session_id": state.get("session_id", "")},
            limit=self._config.graph_findings_limit,
        )

        challenged_hypotheses = {}
        for f in all_findings:
            conclusion = f.get("conclusion_type", "")
            if conclusion in ("inconclusive", "refutes"):
                hid = f.get("hypothesis_id", "")
                if hid and hid not in challenged_hypotheses:
                    challenged_hypotheses[hid] = f

        if not challenged_hypotheses:
            return {}

        # For each challenged hypothesis, let the author respond
        counter_jobs = []
        for hid, critique in challenged_hypotheses.items():
            context = await self._queries.get_hypothesis_context(hid)
            if not context:
                continue

            hypothesis = context["hypothesis"]
            author_id = hypothesis.get("created_by", "")
            if not author_id:
                continue

            prompt = SQUID_COUNTER_RESPONSE.format(
                hypothesis_id=hid,
                hypothesis_text=hypothesis.get("text", ""),
                reviewer_id=critique.get("created_by", "unknown"),
                critique_reasoning=critique.get("text", ""),
            )

            counter_jobs.append(
                {
                    "hypothesis_id": hid,
                    "hypothesis_text": hypothesis.get("text", ""),
                    "author_id": author_id,
                    "reviewer_id": critique.get("created_by", "unknown"),
                    "critique": critique.get("text", ""),
                    "task": self._llm.complete(
                    prompt=prompt,
                    system="You are a research squid defending your hypothesis. "
                    "Respond with one of: ACCEPT (you agree with the critique), "
                    "REBUT (provide counter-evidence), or REQUEST_EXPERIMENT "
                    "(propose a test to settle this).",
                    temperature=self._config.temperature_counter_response,
                    max_tokens=self._config.max_tokens_counter_response,
                    ),
                }
            )

        if not counter_jobs:
            return {}

        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id="debate-system",
            payload={
                "action": "counter_responses_started",
                "challenged_hypotheses": len(counter_jobs),
                "challenged_hypothesis_ids": [
                    job["hypothesis_id"] for job in counter_jobs[:10]
                ],
                "challenged_hypothesis_preview": [
                    {
                        "hypothesis_id": job["hypothesis_id"],
                        "hypothesis_text": job["hypothesis_text"],
                        "author_id": job["author_id"],
                        "reviewer_id": job["reviewer_id"],
                        "critique": job["critique"],
                    }
                    for job in counter_jobs[:6]
                ],
            },
        ))

        completed_responses = 0
        failed_responses = 0
        total_responses = len(counter_jobs)

        async def run_counter(job: dict[str, Any]) -> tuple[dict[str, Any], str | Exception]:
            try:
                response = await job["task"]
                return job, response
            except Exception as exc:  # pragma: no cover - defensive logging
                return job, exc

        for task in asyncio.as_completed([run_counter(job) for job in counter_jobs]):
            job, response = await task
            if isinstance(response, Exception):
                failed_responses += 1
            else:
                completed_responses += 1
                
                # Persist counter-response as a Message
                try:
                    msg = Message(
                        from_agent=job["author_id"],
                        to_agent="",  # Broadcast
                        text=response,
                        message_type=MessageType.EVIDENCE if "REBUT" in response else MessageType.ACKNOWLEDGMENT if "ACCEPT" in response else MessageType.QUESTION,
                        regarding_artifact_id=job["hypothesis_id"],
                        created_by=job["author_id"],
                        session_id=state.get("session_id", ""),
                    )
                    await self._graph.create_message(msg)
                except Exception as e:
                    pass  # Graceful fallback if save fails

            await self._bus.publish(Event(
                event_type=EventType.AGENT_ACTION,
                agent_id="debate-system",
                payload={
                    "action": "counter_response_progress",
                    "completed_responses": completed_responses,
                    "failed_responses": failed_responses,
                    "total_responses": total_responses,
                    "hypothesis_id": job["hypothesis_id"],
                    "hypothesis_text": job["hypothesis_text"],
                    "author_id": job["author_id"],
                    "reviewer_id": job["reviewer_id"],
                },
            ))

        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id="debate-system",
            payload={
                "action": "counter_responses_completed",
                "responses_attempted": total_responses,
                "failed_responses": failed_responses,
            },
        ))

        return {}

    async def _adjudicate(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Make provisional rulings on contested hypotheses.

        For hypotheses that remain contested after counter-responses,
        weigh all evidence and make a ruling: uphold, revise, table
        (needs more evidence), or reject.

        This runs per-cluster, not institute-wide, to keep costs down.
        """
        # Find hypotheses with both supporting and contradicting evidence
        contradictions = await self._queries.get_session_contradictions(
            session_id=state.get("session_id", "")
        )

        if not contradictions:
            return {}

        adjudication_targets = contradictions[
            : self._config.max_adjudications_per_round
        ]

        target_preview = [
            {
                "target_id": c.get("target_id", ""),
                "target_text": c.get("target_text", ""),
                "source_text": c.get("source_text", ""),
            }
            for c in adjudication_targets[:6]
        ]

        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id="debate-system",
            payload={
                "action": "adjudication_started",
                "targets": len(adjudication_targets),
                "target_preview": target_preview,
            },
        ))

        async def adjudicate_one(c: dict[str, Any]) -> dict[str, Any]:
            target_id = c.get("target_id", "")
            if not target_id:
                return {"target_id": "", "target_text": "", "ruling": ""}

            context = await self._queries.get_hypothesis_context(target_id)
            if not context:
                return {
                    "target_id": target_id,
                    "target_text": c.get("target_text", ""),
                    "ruling": "",
                }

            hypothesis = context["hypothesis"]
            await self._bus.publish(Event(
                event_type=EventType.AGENT_ACTION,
                agent_id="adjudicator",
                payload={
                    "action": "adjudicating_hypothesis",
                    "target_id": target_id,
                    "target_text": hypothesis.get("text", ""),
                    "supporters": len(context["supporters"]),
                    "contradictors": len(context["contradictors"]),
                },
            ))
            supporters_text = "\n".join(
                f"- {s.get('text', '')}" for s in context["supporters"]
            ) or "None"
            contradictors_text = "\n".join(
                f"- {c.get('text', '')}" for c in context["contradictors"]
            ) or "None"

            prompt = ADJUDICATOR_PROMPT.format(
                hypothesis_id=target_id,
                hypothesis_text=hypothesis.get("text", ""),
                supporting_evidence=supporters_text,
                contradicting_evidence=contradictors_text,
                experiment_results="None",
            )

            ruling = await self._llm.complete(
                prompt=prompt,
                system="You are a research adjudicator. Make a provisional "
                "ruling based on the evidence. Respond with exactly one of: "
                "UPHOLD, REVISE, TABLE, REJECT — followed by a one-sentence "
                "justification.",
                temperature=self._config.temperature_adjudicator,
                max_tokens=self._config.max_tokens_adjudicator,
            )

            # Parse ruling and update hypothesis status
            ruling_upper = ruling.strip().upper()
            if "REJECT" in ruling_upper:
                await self._graph.update_status(
                    target_id, "refuted", "adjudicator"
                )
            elif "UPHOLD" in ruling_upper:
                await self._graph.update(
                    target_id, {"adjudication_status": "upheld"}
                )
            elif "TABLE" in ruling_upper:
                await self._graph.update(
                    target_id, {"adjudication_status": "tabled"}
                )
            elif "REVISE" in ruling_upper:
                await self._graph.update(
                    target_id, {"adjudication_status": "revised"}
                )

            return {
                "target_id": target_id,
                "target_text": hypothesis.get("text", ""),
                "ruling": ruling.strip(),
            }

        completed_targets = 0
        failed_targets = 0
        total_targets = len(adjudication_targets)
        if adjudication_targets:
            for task in asyncio.as_completed(
                [adjudicate_one(c) for c in adjudication_targets]
            ):
                try:
                    result = await task
                except Exception:  # pragma: no cover - defensive logging
                    failed_targets += 1
                    continue

                completed_targets += 1
                await self._bus.publish(Event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id="debate-system",
                    payload={
                        "action": "adjudication_progress",
                        "completed_targets": completed_targets,
                        "failed_targets": failed_targets,
                        "total_targets": total_targets,
                        "target_id": result.get("target_id", ""),
                        "target_text": result.get("target_text", ""),
                        "ruling": result.get("ruling", ""),
                    },
                ))

        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id="debate-system",
            payload={
                "action": "adjudication_completed",
                "targets": len(adjudication_targets),
                "failed_targets": failed_targets,
            },
        ))

        return {}

    async def _resolve_contradictions(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Identify and flag unresolved contradictions for the next cycle.

        Contradictions that have experiments proposed are left for
        the next research cycle to resolve. Contradictions without
        experiments are flagged as debate topics.
        """
        contradictions = await self._queries.get_session_contradictions(
            session_id=state.get("session_id", "")
        )

        debate_queue = []
        for c in contradictions:
            debate_queue.append({
                "source_id": c.get("source_id", ""),
                "target_id": c.get("target_id", ""),
                "status": "unresolved",
            })

        await self._bus.publish(Event(
            event_type=EventType.DEBATE_COMPLETED,
            payload={
                "iteration": state.get("iteration", 0),
                "contradictions_found": len(contradictions),
                "clusters_used": len(state.get("belief_clusters", [])),
            },
        ))

        return {
            "debate_queue": debate_queue,
        }

    async def _legacy_review(
        self, state: InstituteState
    ) -> dict[str, Any]:
        """
        Fallback: pairwise review when clustering isn't available.

        Used only when there are too few agents for meaningful clusters
        (< 3 agents). Capped to avoid O(N²) explosion.
        """
        agents = state.get("agents", [])
        all_results = []

        for agent in agents:
            if agent["status"] != "active":
                continue

            results = await self._reviewer.review_all_hypotheses(
                reviewer_agent_id=agent["agent_id"],
                exclude_agent=agent["agent_id"],
            )
            all_results.extend(results)

        new_experiments = []
        for r in all_results:
            new_experiments.extend(r.get("experiments_proposed", []))

        return {"pending_experiments": new_experiments}
