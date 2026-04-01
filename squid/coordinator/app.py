"""ResearchSquid Coordinator — FastAPI application factory.

Extended with: SSE telemetry, agent inspector, persona management,
interview system, audit logging, session workflow state.
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from squid.schema.finding import Finding
from squid.schema.experiment import BackendJudgment, ExperimentResult, ExperimentSpec
from squid.schema.session import Session, SessionConfig
from squid.dag.client import DAGClient
from squid.dag.writer import post_finding, write_experiment_spec, post_experiment_result
from squid.dag.reader import get_context, get_frontier, get_paradigm_shifts, get_session_summary
from squid.dag.taxonomy import classify_source_tier
from squid.coordinator.audit import get_audit_logger, EventType
from squid.coordinator.telemetry import get_telemetry
from squid.agents.persona import AgentPersona, create_persona, generate_persona_prompt, PERSONA_TEMPLATES
from squid.dag.persona_store import save_persona, load_persona, load_session_personas
from squid.utils.llm_client import get_llm_client


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


dag_client: Optional[DAGClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dag_client
    dag_client = DAGClient()
    await dag_client.connect()

    # Initialize schema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "dag", "schema.cypher")
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            statements = [s.strip() for s in f.read().split(";") if s.strip() and not s.strip().startswith("--")]
            for stmt in statements:
                try:
                    await dag_client.run(stmt)
                except Exception:
                    pass

    yield
    await dag_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearchSquid Coordinator",
        description="Two-tier autonomous research system",
        version="0.2.0",
        lifespan=lifespan,
    )

    audit = get_audit_logger()
    telemetry = get_telemetry()

    # ================================================================
    # SESSION ENDPOINTS
    # ================================================================

    @app.post("/research", response_model=Session)
    async def start_research(config: SessionConfig):
        session = Session(
            id=gen_id("session"),
            question=config.question,
            modality=config.modality,
            llm_budget_usd=config.llm_budget_usd,
            compute_budget_usd=config.compute_budget_usd,
            agent_count=config.agent_count,
            created_at=datetime.utcnow(),
        )
        await dag_client.run(
            """
            MERGE (s:Session {id: $id})
            SET s.question = $question, s.modality = $modality, s.status = $status,
                s.created_at = $created_at, s.llm_budget_usd = $llm_budget_usd,
                s.compute_budget_usd = $compute_budget_usd, s.llm_spent_usd = 0,
                s.compute_spent_usd = 0
            """,
            id=session.id, question=session.question, modality=session.modality,
            status=session.status, created_at=session.created_at.isoformat(),
            llm_budget_usd=session.llm_budget_usd,
            compute_budget_usd=session.compute_budget_usd,
        )
        audit.log(session.id, EventType.SESSION_STARTED, {
            "question": config.question,
            "modality": config.modality,
            "agent_count": config.agent_count,
        })
        await telemetry.publish(session.id, "session_started", {
            "question": config.question, "modality": config.modality,
        })
        return session

    @app.get("/session/{session_id}")
    async def get_session(session_id: str):
        records = await dag_client.run("MATCH (s:Session {id: $id}) RETURN s", id=session_id)
        if not records:
            raise HTTPException(status_code=404, detail="Session not found")
        return dict(records[0]["s"])

    @app.post("/session/{session_id}/stop")
    async def stop_session(session_id: str):
        await dag_client.run("MATCH (s:Session {id: $id}) SET s.status = 'stopped'", id=session_id)
        audit.log(session_id, EventType.SESSION_COMPLETED, {"status": "stopped"})
        await telemetry.publish(session_id, "session_completed", {"status": "stopped"})
        return {"session_id": session_id, "status": "stopped"}

    @app.post("/session/{session_id}/pause")
    async def pause_session(session_id: str):
        await dag_client.run("MATCH (s:Session {id: $id}) SET s.status = 'paused'", id=session_id)
        audit.log(session_id, EventType.SESSION_PAUSED, {})
        await telemetry.publish(session_id, "session_paused", {})
        return {"session_id": session_id, "status": "paused"}

    @app.post("/session/{session_id}/resume")
    async def resume_session(session_id: str):
        await dag_client.run("MATCH (s:Session {id: $id}) SET s.status = 'active'", id=session_id)
        audit.log(session_id, EventType.SESSION_RESUMED, {})
        await telemetry.publish(session_id, "session_resumed", {})
        return {"session_id": session_id, "status": "active"}

    @app.get("/session/{session_id}/summary")
    async def summary(session_id: str):
        return {"summary": await get_session_summary(dag_client, session_id)}

    @app.get("/session/{session_id}/dag")
    async def get_dag(session_id: str):
        nodes = await dag_client.run(
            "MATCH (n) WHERE n.session_id = $sid RETURN labels(n) as labels, properties(n) as props",
            sid=session_id,
        )
        edges = await dag_client.run(
            "MATCH (a)-[r]->(b) WHERE a.session_id = $sid OR b.session_id = $sid RETURN a.id as src, type(r) as type, b.id as tgt, properties(r) as props",
            sid=session_id,
        )
        return {
            "nodes": [{"labels": r["labels"], "props": dict(r["props"])} for r in nodes],
            "edges": [{"src": r["src"], "type": r["type"], "tgt": r["tgt"], "props": dict(r["props"])} for r in edges],
        }

    # ================================================================
    # SESSION STATUS (rich — agents, experiments, budget, workflow step)
    # ================================================================

    @app.get("/session/{session_id}/status")
    async def session_status(session_id: str):
        """Rich session status for UI workflow display."""
        # Agents
        agent_records = await dag_client.run(
            "MATCH (a:Agent {session_id: $sid}) RETURN a", sid=session_id
        )
        agents = [dict(r["a"]) for r in agent_records]

        # Findings
        finding_records = await dag_client.run(
            "MATCH (f:Finding {session_id: $sid}) RETURN f", sid=session_id
        )
        findings = [dict(r["f"]) for r in finding_records]

        # Experiments
        exp_records = await dag_client.run(
            "MATCH (e:Experiment {session_id: $sid}) RETURN e", sid=session_id
        )
        experiments = [dict(r["e"]) for r in exp_records]

        # Run records
        run_records = await dag_client.run(
            "MATCH (r:ExperimentRun {session_id: $sid}) RETURN r", sid=session_id
        )
        runs = [dict(r["r"]) for r in run_records]

        # Session
        session_records = await dag_client.run(
            "MATCH (s:Session {id: $id}) RETURN s", id=session_id
        )
        session_data = dict(session_records[0]["s"]) if session_records else {}

        # Contradictions
        contradictions = [
            f for f in findings
            if f.get("relation_type") == "CONTRADICTS" and f.get("status") == "active"
        ]

        # Determine workflow step
        workflow_step = _determine_workflow_step(findings, experiments, runs, session_data)

        return {
            "session_id": session_id,
            "workflow_step": workflow_step,
            "question": session_data.get("question", ""),
            "status": session_data.get("status", "unknown"),
            "agents": {
                "total": len(agents),
                "active": len([a for a in agents if a.get("status") == "researching"]),
                "sleeping": len([a for a in agents if a.get("status") == "sleeping"]),
            },
            "findings": {
                "total": len(findings),
                "active": len([f for f in findings if f.get("status") == "active"]),
                "with_verification": len([f for f in findings if f.get("has_numerical_verification")]),
            },
            "experiments": {
                "total": len(experiments),
                "pending": len([e for e in experiments if e.get("status") == "pending"]),
                "running": len([e for e in experiments if e.get("status") == "running"]),
                "completed": len([e for e in experiments if e.get("status") == "completed"]),
                "failed": len([e for e in experiments if e.get("status") == "failed"]),
            },
            "runs": {
                "total": len(runs),
                "passed": len([r for r in runs if r.get("passed_constraints")]),
            },
            "contradictions": len(contradictions),
            "budget": {
                "llm_spent": session_data.get("llm_spent_usd", 0),
                "llm_budget": session_data.get("llm_budget_usd", 0),
                "compute_spent": session_data.get("compute_spent_usd", 0),
                "compute_budget": session_data.get("compute_budget_usd", 0),
            },
            "best_answer": _get_best_answer_label(findings),
        }

    def _determine_workflow_step(findings, experiments, runs, session_data):
        if session_data.get("status") in ("stopped", "completed"):
            return 6  # Report
        if runs:
            return 5  # Clusters/Debates
        if experiments:
            return 4  # Experiment Queue
        if findings:
            return 3  # Agent Search
        if session_data.get("status") == "active":
            return 1  # Question/Goal
        return 0

    def _get_best_answer_label(findings):
        active = [f for f in findings if f.get("status") == "active"]
        if not active:
            return {"label": "Insufficient evidence", "confidence": 0}
        best = max(active, key=lambda f: f.get("confidence", 0))
        conf = best.get("confidence", 0)
        if conf >= 0.85:
            label = "High confidence"
        elif conf >= 0.65:
            label = "Moderately confident"
        elif conf >= 0.45:
            label = "Tentative"
        elif conf >= 0.25:
            label = "Weak evidence"
        else:
            label = "Insufficient evidence"
        return {"label": label, "confidence": conf, "claim": best.get("claim", "")}

    # ================================================================
    # SSE TELEMETRY STREAM
    # ================================================================

    @app.get("/session/{session_id}/stream")
    async def session_stream(session_id: str):
        """SSE stream of session events."""
        return StreamingResponse(
            telemetry.sse_stream(session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ================================================================
    # AGENT INSPECTOR
    # ================================================================

    @app.get("/session/{session_id}/agent/{agent_id}")
    async def inspect_agent(session_id: str, agent_id: str):
        """Inspect one agent's full state."""
        # Agent node
        agent_records = await dag_client.run(
            "MATCH (a:Agent {id: $aid, session_id: $sid}) RETURN a",
            aid=agent_id, sid=session_id,
        )
        agent_data = dict(agent_records[0]["a"]) if agent_records else {"id": agent_id}

        # Agent's findings
        finding_records = await dag_client.run(
            "MATCH (f:Finding {agent_id: $aid, session_id: $sid}) RETURN f ORDER BY f.created_at DESC LIMIT 20",
            aid=agent_id, sid=session_id,
        )
        findings = [dict(r["f"]) for r in finding_records]

        # Agent's experiments
        exp_records = await dag_client.run(
            "MATCH (e:Experiment {submitted_by: $aid, session_id: $sid}) RETURN e ORDER BY e.submitted_at DESC LIMIT 10",
            aid=agent_id, sid=session_id,
        )
        experiments = [dict(r["e"]) for r in exp_records]

        # Persona (load from Neo4j)
        persona = await load_persona(dag_client, session_id, agent_id)

        # Cluster membership
        cluster_records = await dag_client.run(
            "MATCH (c:Cluster)<-[:BELONGS_TO]-(f:Finding {agent_id: $aid, session_id: $sid}) RETURN DISTINCT c",
            aid=agent_id, sid=session_id,
        )
        clusters = [dict(r["c"]) for r in cluster_records]

        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "agent_data": agent_data,
            "persona": persona.model_dump() if persona else None,
            "findings_count": len(findings),
            "findings_recent": findings[:5],
            "experiments_count": len(experiments),
            "experiments_recent": experiments[:5],
            "clusters": clusters,
            "current_hypothesis": _extract_current_hypothesis(findings),
        }

    def _extract_current_hypothesis(findings):
        """Extract the agent's current primary hypothesis from their findings."""
        active = [f for f in findings if f.get("status") == "active"]
        if not active:
            return None
        best = max(active, key=lambda f: f.get("confidence", 0))
        return {
            "claim": best.get("claim"),
            "confidence": best.get("confidence"),
            "evidence_type": best.get("evidence_type"),
        }

    # ================================================================
    # PERSONA MANAGEMENT
    # ================================================================

    @app.get("/personas/templates")
    async def list_persona_templates():
        """List available persona templates."""
        return {
            "templates": {
                name: {k: v for k, v in data.items()}
                for name, data in PERSONA_TEMPLATES.items()
            }
        }

    @app.post("/session/{session_id}/agent/{agent_id}/persona")
    async def set_agent_persona(
        session_id: str,
        agent_id: str,
        template: Optional[str] = None,
        specialty: Optional[str] = None,
        skepticism_level: Optional[float] = None,
        preferred_evidence_types: Optional[List[str]] = None,
        contradiction_aggressiveness: Optional[float] = None,
        source_strictness: Optional[float] = None,
        experiment_appetite: Optional[float] = None,
        reporting_style: Optional[str] = None,
        model_tier: Optional[str] = None,
    ):
        """Set or update agent persona. Only strategy fields are editable."""
        # Load existing from Neo4j
        existing = await load_persona(dag_client, session_id, agent_id)

        if existing:
            # Revision: update editable fields, keep revision history
            revision_entry = {
                "revision": existing.revision,
                "timestamp": datetime.utcnow().isoformat(),
                "changes": {},
            }
            persona = existing
            persona.revision += 1
            persona.updated_at = datetime.utcnow()

            if specialty is not None:
                revision_entry["changes"]["specialty"] = (persona.specialty, specialty)
                persona.specialty = specialty
            if skepticism_level is not None:
                revision_entry["changes"]["skepticism_level"] = (persona.skepticism_level, skepticism_level)
                persona.skepticism_level = skepticism_level
            if preferred_evidence_types is not None:
                revision_entry["changes"]["preferred_evidence_types"] = (persona.preferred_evidence_types, preferred_evidence_types)
                persona.preferred_evidence_types = preferred_evidence_types
            if contradiction_aggressiveness is not None:
                revision_entry["changes"]["contradiction_aggressiveness"] = (persona.contradiction_aggressiveness, contradiction_aggressiveness)
                persona.contradiction_aggressiveness = contradiction_aggressiveness
            if source_strictness is not None:
                revision_entry["changes"]["source_strictness"] = (persona.source_strictness, source_strictness)
                persona.source_strictness = source_strictness
            if experiment_appetite is not None:
                revision_entry["changes"]["experiment_appetite"] = (persona.experiment_appetite, experiment_appetite)
                persona.experiment_appetite = experiment_appetite
            if reporting_style is not None:
                revision_entry["changes"]["reporting_style"] = (persona.reporting_style, reporting_style)
                persona.reporting_style = reporting_style
            if model_tier is not None:
                revision_entry["changes"]["model_tier"] = (persona.model_tier, model_tier)
                persona.model_tier = model_tier

            persona.revision_history.append(revision_entry)
        else:
            # New persona
            persona = create_persona(agent_id, session_id, template=template)
            if specialty is not None:
                persona.specialty = specialty
            if skepticism_level is not None:
                persona.skepticism_level = skepticism_level
            if preferred_evidence_types is not None:
                persona.preferred_evidence_types = preferred_evidence_types
            if contradiction_aggressiveness is not None:
                persona.contradiction_aggressiveness = contradiction_aggressiveness
            if source_strictness is not None:
                persona.source_strictness = source_strictness
            if experiment_appetite is not None:
                persona.experiment_appetite = experiment_appetite
            if reporting_style is not None:
                persona.reporting_style = reporting_style
            if model_tier is not None:
                persona.model_tier = model_tier

        # Persist to Neo4j
        await save_persona(dag_client, persona)

        audit.log(session_id, EventType.PERSONA_REVISION_APPLIED, {
            "agent_id": agent_id,
            "revision": persona.revision,
            "persona_id": persona.id,
        }, agent_id=agent_id)
        await telemetry.publish(session_id, "persona_revision_applied", {
            "agent_id": agent_id, "revision": persona.revision,
        }, agent_id=agent_id)

        return {"persona": persona.model_dump(), "prompt_addition": generate_persona_prompt(persona)}

    @app.get("/session/{session_id}/agent/{agent_id}/persona")
    async def get_agent_persona(session_id: str, agent_id: str):
        """Get agent's current persona from Neo4j."""
        persona = await load_persona(dag_client, session_id, agent_id)
        if not persona:
            return {"persona": None, "prompt_addition": None}
        return {"persona": persona.model_dump(), "prompt_addition": generate_persona_prompt(persona)}

    # ================================================================
    # AGENT INTERVIEW
    # ================================================================

    class InterviewRequest(BaseModel):
        agent_id: str
        prompt: str
        max_context_findings: int = 10

    class BatchInterviewRequest(BaseModel):
        interviews: List[InterviewRequest]
        shared_prompt: Optional[str] = None

    @app.post("/session/{session_id}/interview")
    async def interview_agent(session_id: str, req: InterviewRequest):
        """Interview a single agent — read-only, grounded in their history. Uses LLM."""
        audit.log(session_id, EventType.INTERVIEW_STARTED, {
            "agent_id": req.agent_id, "prompt": req.prompt,
        }, agent_id=req.agent_id)

        # Gather agent context
        finding_records = await dag_client.run(
            "MATCH (f:Finding {agent_id: $aid, session_id: $sid}) RETURN f ORDER BY f.confidence DESC LIMIT $limit",
            aid=req.agent_id, sid=session_id, limit=req.max_context_findings,
        )
        findings = [dict(r["f"]) for r in finding_records]

        exp_records = await dag_client.run(
            "MATCH (e:Experiment {submitted_by: $aid, session_id: $sid}) RETURN e ORDER BY e.submitted_at DESC LIMIT 5",
            aid=req.agent_id, sid=session_id,
        )
        experiments = [dict(r["e"]) for r in exp_records]

        persona = await load_persona(dag_client, session_id, req.agent_id)

        # Build grounded context for LLM
        findings_text = "\n".join(
            f"- [{f.get('confidence', 0):.2f}] {f.get('claim', 'N/A')} (evidence: {f.get('evidence_type', 'N/A')}, tier: {f.get('min_source_tier', '?')})"
            for f in findings[:10]
        ) or "No findings posted yet."

        experiments_text = "\n".join(
            f"- {e.get('goal', 'N/A')} (status: {e.get('status', 'N/A')}, backend: {e.get('backend_type', 'N/A')})"
            for e in experiments
        ) or "No experiments submitted yet."

        persona_text = f"Specialty: {persona.specialty}, Skepticism: {persona.skepticism_level:.0%}" if persona else "No persona set."

        hypothesis = _extract_current_hypothesis(findings)
        hypothesis_text = f"{hypothesis['claim']} (confidence: {hypothesis['confidence']:.2f})" if hypothesis else "No clear hypothesis yet."

        # Call LLM for grounded response
        llm = get_llm_client()
        system_prompt = (
            "You are a research agent being interviewed about your work. "
            "Answer based ONLY on the findings, experiments, and persona provided. "
            "Do not invent new claims. If you don't have evidence for something, say so. "
            "Be honest about uncertainty. Reference specific findings when relevant."
        )

        user_prompt = f"""## Your Research Context

**Current hypothesis:** {hypothesis_text}

**Your persona:** {persona_text}

**Your findings:**
{findings_text}

**Your experiments:**
{experiments_text}

## Interview Question

{req.prompt}

Answer as this agent would, grounded in the evidence above."""

        try:
            grounded_response = llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=1024,
            )
        except Exception as e:
            grounded_response = f"LLM call failed: {str(e)}. Context provided for manual review."

        response = {
            "agent_id": req.agent_id,
            "prompt": req.prompt,
            "grounded_response": grounded_response,
            "context_summary": {
                "findings_count": len(findings),
                "experiments_count": len(experiments),
                "current_hypothesis": hypothesis,
                "persona_specialty": persona.specialty if persona else "general",
            },
        }

        audit.log(session_id, EventType.INTERVIEW_COMPLETED, {
            "agent_id": req.agent_id,
            "findings_used": len(findings),
            "response_length": len(grounded_response),
        }, agent_id=req.agent_id)

        return response

    @app.post("/session/{session_id}/interview/batch")
    async def interview_batch(session_id: str, req: BatchInterviewRequest):
        """Interview multiple agents."""
        results = {}
        for interview in req.interviews:
            prompt = req.shared_prompt or interview.prompt
            single_req = InterviewRequest(
                agent_id=interview.agent_id,
                prompt=prompt,
                max_context_findings=interview.max_context_findings,
            )
            result = await interview_agent(session_id, single_req)
            results[interview.agent_id] = result
        return {"interviews": results, "count": len(results)}

    @app.post("/session/{session_id}/interview/all")
    async def interview_all(session_id: str, prompt: str, max_agents: int = 25):
        """Interview all agents with the same question."""
        agent_records = await dag_client.run(
            "MATCH (a:Agent {session_id: $sid}) RETURN a.id as agent_id LIMIT $limit",
            sid=session_id, limit=max_agents,
        )
        agent_ids = [r["agent_id"] for r in agent_records]

        req = BatchInterviewRequest(
            interviews=[InterviewRequest(agent_id=aid, prompt=prompt) for aid in agent_ids],
            shared_prompt=prompt,
        )
        return await interview_batch(session_id, req)

    # ================================================================
    # INTERNAL ENDPOINTS (Tier-1 agents)
    # ================================================================

    class FindingRequest(BaseModel):
        claim: str
        confidence: float
        confidence_rationale: str
        evidence_type: str
        source_urls: List[str] = []
        numerical_verification_ran: bool = False
        experiment_run_id: Optional[str] = None
        relates_to: Optional[str] = None
        relation_type: Optional[str] = None
        counter_claim: Optional[str] = None
        session_id: str
        agent_id: str

    @app.post("/internal/finding")
    async def post_finding_endpoint(req: FindingRequest):
        source_tiers = [classify_source_tier(url)[0] for url in req.source_urls]
        finding = Finding(
            id=gen_id("f"),
            session_id=req.session_id,
            agent_id=req.agent_id,
            claim=req.claim,
            confidence=req.confidence,
            confidence_rationale=req.confidence_rationale,
            evidence_type=req.evidence_type,  # type: ignore
            source_urls=req.source_urls,
            source_tiers=source_tiers,
            min_source_tier=min(source_tiers) if source_tiers else 4,
            has_numerical_verification=req.numerical_verification_ran,
            experiment_run_id=req.experiment_run_id,
            relates_to=req.relates_to,
            relation_type=req.relation_type,  # type: ignore
            counter_claim=req.counter_claim,
        )
        fid = await post_finding(dag_client, finding)

        audit.log(req.session_id, EventType.FINDING_POSTED, {
            "finding_id": fid, "claim": req.claim, "confidence": req.confidence,
        }, agent_id=req.agent_id)
        await telemetry.publish(req.session_id, "finding_posted", {
            "finding_id": fid, "claim": req.claim, "confidence": req.confidence,
            "agent_id": req.agent_id,
        }, agent_id=req.agent_id)

        if req.relation_type == "CONTRADICTS":
            audit.log(req.session_id, EventType.CONTRADICTION_OPENED, {
                "finding_id": fid, "target": req.relates_to, "counter_claim": req.counter_claim,
            }, agent_id=req.agent_id)
            await telemetry.publish(req.session_id, "contradiction_opened", {
                "finding_id": fid, "counter_claim": req.counter_claim,
            }, agent_id=req.agent_id)

        return {"finding_id": fid, "status": "created"}

    @app.post("/internal/experiment")
    async def submit_experiment(spec: ExperimentSpec, session_id: str, agent_id: str):
        spec.id = gen_id("spec")
        spec.session_id = session_id
        spec.submitted_by = agent_id
        spec.submitted_at = datetime.utcnow()
        sid = await write_experiment_spec(dag_client, spec)

        audit.log(session_id, EventType.EXPERIMENT_SUBMITTED, {
            "spec_id": sid, "executor_type": spec.backend_type,
        }, agent_id=agent_id)
        await telemetry.publish(session_id, "experiment_submitted", {
            "spec_id": sid, "executor_type": spec.backend_type, "agent_id": agent_id,
        }, agent_id=agent_id)

        return {"spec_id": sid, "status": "queued"}

    @app.get("/internal/context/{agent_id}")
    async def agent_context(agent_id: str, session_id: str):
        return await get_context(dag_client, session_id, agent_id)

    # ================================================================
    # BACKEND-FACING
    # ================================================================

    @app.get("/internal/experiments/queue")
    async def experiment_queue(session_id: Optional[str] = None):
        if session_id:
            from squid.dag.reader import get_pending_experiments
            exps = await get_pending_experiments(dag_client, session_id)
        else:
            records = await dag_client.run(
                "MATCH (e:Experiment {status: 'pending'}) RETURN e ORDER BY e.submitted_at ASC LIMIT 10"
            )
            exps = [dict(r["e"]) for r in records]
        return {"experiments": exps}

    @app.post("/internal/experiments/{spec_id}/result")
    async def post_result(spec_id: str, result: ExperimentResult):
        result.spec_id = spec_id
        rid = await post_experiment_result(dag_client, result)

        status_event = EventType.EXPERIMENT_COMPLETED if result.status == "completed" else EventType.EXPERIMENT_FAILED
        audit.log(result.session_id, status_event, {
            "run_id": rid, "spec_id": spec_id, "status": result.status.value,
        })
        await telemetry.publish(result.session_id, f"experiment_{result.status.value}", {
            "run_id": rid, "spec_id": spec_id,
        })

        return {"run_id": rid, "status": result.status}

    # ================================================================
    # REPORT GENERATION
    # ================================================================

    @app.post("/session/{session_id}/report/plan")
    async def plan_report(session_id: str):
        """Plan report outline using ReportAgent."""
        from squid.coordinator.report_agent import ReportAgent
        agent = ReportAgent(dag_client, session_id)
        session_records = await dag_client.run("MATCH (s:Session {id: $id}) RETURN s", id=session_id)
        question = session_records[0]["s"]["question"] if session_records else ""
        outline = await agent.plan_outline(question)
        return {"outline": outline, "audit_log": agent.audit_log}

    @app.get("/session/{session_id}/report")
    async def generate_report(session_id: str):
        """Generate full report using tool-using ReportAgent."""
        from squid.coordinator.report_agent import ReportAgent
        agent = ReportAgent(dag_client, session_id)

        session_records = await dag_client.run("MATCH (s:Session {id: $id}) RETURN s", id=session_id)
        question = session_records[0]["s"]["question"] if session_records else ""

        outline = await agent.plan_outline(question)
        findings = await get_frontier(dag_client, session_id)

        sections = []
        for section in outline:
            audit.log(session_id, EventType.REPORT_PROGRESS, {
                "section": section["title"], "status": "generating",
            })
            await telemetry.publish(session_id, "report_progress", {
                "section": section["title"], "status": "generating",
            })

            content = await agent.generate_section(section, findings)
            sections.append({"title": section["title"], "content": content})

        report = "\n\n".join(s["content"] for s in sections)

        audit.log(session_id, EventType.REPORT_PROGRESS, {"status": "complete"})
        await telemetry.publish(session_id, "report_progress", {"status": "complete"})

        return {
            "report": report,
            "sections": sections,
            "audit_log": agent.audit_log,
        }

    # ================================================================
    # AUDIT LOG
    # ================================================================

    @app.get("/session/{session_id}/audit")
    async def get_audit_log(session_id: str, limit: int = 100, offset: int = 0):
        """Read session audit log."""
        events = audit.read_log(session_id, limit=limit, offset=offset)
        return {"events": events, "count": len(events)}

    # ================================================================
    # HEALTH
    # ================================================================

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "research-squid-coordinator", "version": "0.2.0"}

    return app


app = create_app()
