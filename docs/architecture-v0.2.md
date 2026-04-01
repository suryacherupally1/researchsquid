# HiveResearch v0.2 — Architecture Updates

## What Changed (MiroFish-Inspired Improvements)

### P1: Session Workflow + Real-Time Telemetry

**New:** Linear 7-step workflow visible to users at all times.

Workflow steps:
1. Question / Goal — user provides research question
2. Setup — context, modality, backend configuration
3. Agent Search — agents searching, posting findings
4. Experiments — ExperimentSpec queue + execution runs
5. Clusters / Debates — epistemic clusters, contradictions
6. Report — tool-using report generation
7. Interview / Follow-up — post-research agent interviews

**New:** SSE telemetry stream at `GET /session/{id}/stream`

Event types:
- `session_started`, `session_paused`, `session_resumed`, `session_completed`
- `agent_spawned`, `agent_status_changed`, `agent_hypothesis_updated`
- `finding_posted`
- `contradiction_opened`
- `experiment_submitted`, `experiment_started`, `experiment_completed`, `experiment_failed`
- `cluster_membership_changed`
- `budget_updated`
- `report_progress`
- `persona_revision_applied`
- `interview_started`, `interview_completed`

**New:** Rich session status endpoint at `GET /session/{id}/status`

Returns workflow step, agent counts (active/sleeping), finding counts, experiment counts, contradictions, budget summary, and current best answer label.

### P2: Agent Inspector

**New:** `GET /session/{id}/agent/{agent_id}`

Returns:
- Agent data (from Neo4j)
- Current persona
- Recent findings (last 5)
- Recent experiments (last 5)
- Cluster membership
- Current hypothesis (extracted from highest-confidence finding)

### P3: Persona Revision System

**New:** `hive/agents/persona.py`

8 research-specialization templates:
- `pharmacology` — high source strictness, moderate skepticism
- `medicinal_chemistry` — high experiment appetite
- `methods_skeptic` — maximum skepticism, empirical-only evidence
- `cost_analysis` — concise reporting, moderate experiments
- `optimization` — high experiment appetite, computational focus
- `simulation_reliability` — high skepticism, computational + empirical
- `literature_critic` — high source strictness, low experiments
- `devil_advocate` — maximum contradiction aggressiveness

Editable fields (strategy only):
- specialty, skepticism_level, preferred_evidence_types
- contradiction_aggressiveness, source_strictness, experiment_appetite, reporting_style

NOT editable (safety-critical):
- tool allowlists, backend permissions, safety rules, budget limits, system prompts

**New endpoints:**
- `GET /personas/templates` — list available templates
- `POST /session/{id}/agent/{id}/persona` — set/update persona
- `GET /session/{id}/agent/{id}/persona` — get current persona

Every revision creates a versioned entry in revision_history. Old findings remain attributable to old persona revision.

### P4: Agent Interview System

**New endpoints:**
- `POST /session/{id}/interview` — interview single agent (read-only, grounded)
- `POST /session/{id}/interview/batch` — interview multiple agents
- `POST /session/{id}/interview/all` — interview all agents with same question

Interviews are read-only. They read the agent's findings, experiments, persona, and cluster history. They do NOT modify agent behavior or inject new instructions.

### P5: Tool-Using Report Agent

**New:** `hive/coordinator/report_agent.py`

ReportAgent uses ReACT-style tool calling:
- `read_finding(id)` — read specific finding
- `search_findings(query)` — search by claim text
- `get_experiment_run(id)` — read experiment details
- `get_cluster_details(id)` — read cluster with members
- `get_agent_findings(agent_id)` — read agent's findings

Flow: Plan outline → Gather evidence → Verify claims → Write sections → Reflect

Preserves honest-contract output:
- Calibrated confidence labels
- Limitations section (mandatory)
- Minority positions preserved
- Insufficient-evidence terminal state

**New endpoints:**
- `POST /session/{id}/report/plan` — plan report outline
- `GET /session/{id}/report` — generate full report

### P6: Audit Logging

**New:** `hive/coordinator/audit.py`

Structured JSONL audit log per session. Events:
- Session lifecycle, agent lifecycle, experiment lifecycle
- Report generation progress, persona revisions, interviews

**New endpoint:** `GET /session/{id}/audit?limit=100&offset=0`

### P7: Developer UX

**New:** `scripts/setup.sh` — one-command local dev setup

---

## What Was Strictly NOT Taken from MiroFish

| MiroFish Pattern | Rejected | Why |
|---|---|---|
| File-based IPC | ✅ Rejected | Fragile. All communication stays HTTP/event-based |
| Zep Cloud for core graph | ✅ Rejected | Neo4j remains source of truth, self-hosted |
| No disagreement between agents | ✅ Rejected | CONTRADICTS + counter_claim preserved |
| No source quality controls | ✅ Rejected | 4-tier taxonomy, numerical verification preserved |
| Polling-everything architecture | ✅ Rejected | SSE streaming preferred |
| Opaque agent reasoning | ✅ Rejected | Agent inspector exposes reasoning context |
| Social simulation framing | ✅ Rejected | HiveResearch asks "what is true?", not "what will people do?" |
| Theatrical persona roleplay | ✅ Rejected | Personas are research specializations, not characters |

---

## New API Endpoints Summary

| Method | Path | Purpose |
|---|---|---|
| GET | /session/{id}/status | Rich session workflow status |
| GET | /session/{id}/stream | SSE telemetry stream |
| GET | /session/{id}/agent/{aid} | Agent inspector |
| GET | /personas/templates | List persona templates |
| POST | /session/{id}/agent/{aid}/persona | Set/update persona |
| GET | /session/{id}/agent/{aid}/persona | Get persona |
| POST | /session/{id}/interview | Interview single agent |
| POST | /session/{id}/interview/batch | Interview multiple agents |
| POST | /session/{id}/interview/all | Interview all agents |
| POST | /session/{id}/report/plan | Plan report outline |
| GET | /session/{id}/report | Generate full report |
| GET | /session/{id}/audit | Read audit log |
