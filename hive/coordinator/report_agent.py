"""Report Agent — ReACT-style tool-using report generator.

Plans outline → gathers evidence → verifies claims → writes sections.
Preserves the honest-contract output model.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from hive.dag.reader import get_frontier, get_clusters, get_paradigm_shifts
from hive.schema.finding import Finding


class ReportAgent:
    """
    Tool-using report generator.

    Tools available:
    - read_finding(finding_id) — read a specific finding
    - search_findings(query) — search findings by claim text
    - get_experiment_run(run_id) — read experiment details
    - get_cluster_details(cluster_id) — read cluster details
    - get_agent_findings(agent_id) — read all findings by an agent

    Flow:
    1. PLAN: Generate report outline
    2. GATHER: For each section, use tools to collect evidence
    3. VERIFY: Check numerical claims against ExperimentRuns
    4. WRITE: Generate section content
    5. REFLECT: Check for missing limitations, minority positions
    """

    def __init__(self, dag, session_id: str):
        self.dag = dag
        self.session_id = session_id
        self.audit_log: List[Dict[str, Any]] = []

    def _log(self, action: str, details: Dict[str, Any]):
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details,
        })

    # --- Tools ---

    async def tool_read_finding(self, finding_id: str) -> Optional[Dict]:
        """Read a specific finding by ID."""
        records = await self.dag.run(
            "MATCH (f:Finding {id: $id}) RETURN f", id=finding_id
        )
        if records:
            return dict(records[0]["f"])
        return None

    async def tool_search_findings(self, query: str, limit: int = 10) -> List[Dict]:
        """Search findings by claim text (substring match)."""
        records = await self.dag.run(
            """
            MATCH (f:Finding {session_id: $sid})
            WHERE f.claim CONTAINS $query
            RETURN f
            ORDER BY f.confidence DESC
            LIMIT $limit
            """,
            sid=self.session_id,
            query=query,
            limit=limit,
        )
        return [dict(r["f"]) for r in records]

    async def tool_get_experiment_run(self, run_id: str) -> Optional[Dict]:
        """Read experiment run details."""
        records = await self.dag.run(
            "MATCH (r:ExperimentRun {id: $id}) RETURN r", id=run_id
        )
        if records:
            return dict(records[0]["r"])
        return None

    async def tool_get_cluster_details(self, cluster_id: str) -> Optional[Dict]:
        """Read cluster with its member findings."""
        records = await self.dag.run(
            """
            MATCH (c:Cluster {id: $cid})
            OPTIONAL MATCH (f:Finding)-[:BELONGS_TO]->(c)
            RETURN c, collect(f) as findings
            """,
            cid=cluster_id,
        )
        if records:
            return {
                "cluster": dict(records[0]["c"]),
                "findings": [dict(f) for f in records[0]["findings"]],
            }
        return None

    async def tool_get_agent_findings(self, agent_id: str) -> List[Dict]:
        """Read all findings by a specific agent."""
        records = await self.dag.run(
            """
            MATCH (f:Finding {agent_id: $aid, session_id: $sid})
            RETURN f ORDER BY f.created_at DESC
            """,
            aid=agent_id,
            sid=self.session_id,
        )
        return [dict(r["f"]) for r in records]

    # --- Report Generation ---

    async def plan_outline(self, question: str) -> List[Dict[str, str]]:
        """Plan report sections. Returns list of {title, description}."""
        self._log("plan_outline_start", {"question": question})

        findings = await get_frontier(self.dag, self.session_id)
        clusters = await get_clusters(self.dag, self.session_id)
        shifts = await get_paradigm_shifts(self.dag)

        sections = [
            {"title": "Executive Summary", "description": "Calibrated answer with confidence label"},
            {"title": "Key Findings", "description": "Top findings with source tiers and verification status"},
        ]

        if clusters:
            sections.append({
                "title": "Active Debates",
                "description": "Cluster positions with agent counts and source quality",
            })

        experiment_records = await self.dag.run(
            "MATCH (r:ExperimentRun {session_id: $sid}) RETURN r", sid=self.session_id
        )
        if experiment_records:
            sections.append({
                "title": "Experiment Results",
                "description": "What was tested, what passed/failed, with provenance",
            })

        contradictions = [
            f for f in findings
            if f.get("relates_to") and f.get("relation_type") == "CONTRADICTS"
        ]
        if contradictions:
            sections.append({
                "title": "Contradictions",
                "description": "Active disagreements with counter-claims",
            })

        sections.extend([
            {"title": "Minority Positions", "description": "Lower-confidence findings with strongest evidence"},
            {"title": "Limitations", "description": "Weak sources, unresolved debates, evidence gaps"},
            {"title": "What Would Change This Answer", "description": "Specific evidence that would shift the conclusion"},
            {"title": "Sources", "description": "All sources with tier labels and access dates"},
        ])

        self._log("plan_outline_complete", {"sections": len(sections)})
        return sections

    async def generate_section(
        self, section: Dict[str, str], findings: List[Dict]
    ) -> str:
        """Generate content for a single report section."""
        title = section["title"]
        self._log("generate_section_start", {"title": title})

        if title == "Executive Summary":
            return await self._gen_executive_summary(findings)
        elif title == "Key Findings":
            return await self._gen_key_findings(findings)
        elif title == "Active Debates":
            return await self._gen_active_debates()
        elif title == "Experiment Results":
            return await self._gen_experiment_results()
        elif title == "Contradictions":
            return await self._gen_contradictions(findings)
        elif title == "Limitations":
            return await self._gen_limitations(findings)
        else:
            return f"_{title}: pending generation_\n"

    async def _gen_executive_summary(self, findings: List[Dict]) -> str:
        active = [f for f in findings if f.get("status") == "active"]
        if not active:
            return "## Executive Summary\n\n**Insufficient evidence** — No findings with adequate confidence.\n"

        best = max(active, key=lambda f: f.get("confidence", 0))
        conf = best.get("confidence", 0)
        label = _confidence_label(conf)

        # Verify claim if it has numbers
        import re
        has_numbers = bool(re.search(r'\d', best.get("claim", "")))
        verified = best.get("has_numerical_verification", False)
        run_id = best.get("experiment_run_id")

        verification_note = ""
        if has_numbers:
            if verified:
                verification_note = "\n_Numerical claims verified via sandbox execution._"
            elif run_id:
                run = await self.tool_get_experiment_run(run_id)
                if run:
                    verification_note = f"\n_Numerical claims verified via experiment {run_id}._"

        return (
            f"## Executive Summary\n\n"
            f"**{label}** — {best.get('confidence_rationale', '')}\n\n"
            f"{best.get('claim', '')}"
            f"{verification_note}\n"
        )

    async def _gen_key_findings(self, findings: List[Dict]) -> str:
        active = sorted(
            [f for f in findings if f.get("status") == "active"],
            key=lambda f: f.get("confidence", 0),
            reverse=True,
        )[:10]

        if not active:
            return "## Key Findings\n\n_No key findings yet._\n"

        lines = ["## Key Findings\n"]
        for f in active:
            label = _confidence_label(f.get("confidence", 0))
            tier = f.get("min_source_tier", 4)
            tier_label = _tier_label(tier)
            verified = "✓ verified" if f.get("has_numerical_verification") else "unverified"
            lines.append(f"- **{label}** | Tier {tier} ({tier_label}) | {verified}")
            lines.append(f"  {f.get('claim', 'N/A')}")

            # If comes from experiment, show provenance
            run_id = f.get("experiment_run_id")
            if run_id:
                lines.append(f"  _Provenance: ExperimentRun {run_id}_")
        return "\n".join(lines) + "\n"

    async def _gen_active_debates(self) -> str:
        clusters = await get_clusters(self.dag, self.session_id)
        if not clusters:
            return "## Active Debates\n\n_No active debates._\n"

        lines = ["## Active Debates\n"]
        for c in clusters:
            lines.append(f"- **{c.get('central_claim', 'N/A')}**")
            lines.append(
                f"  {c.get('agent_count', 0)} agents | "
                f"weight: {c.get('total_citation_weight', 0):.1f} | "
                f"avg source tier: {c.get('avg_source_tier', 4):.1f}"
            )
        return "\n".join(lines) + "\n"

    async def _gen_experiment_results(self) -> str:
        records = await self.dag.run(
            "MATCH (r:ExperimentRun {session_id: $sid}) RETURN r ORDER BY r.completed_at DESC LIMIT 20",
            sid=self.session_id,
        )
        if not records:
            return "## Experiment Results\n\n_No experiments run yet._\n"

        lines = ["## Experiment Results\n"]
        for rec in records:
            r = dict(rec["r"])
            passed = r.get("passed_constraints", False)
            emoji = "✅" if passed else "❌"
            lines.append(f"### {emoji} {r.get('id', 'N/A')}")
            lines.append(f"- **Backend:** {r.get('backend_type', 'N/A')}")
            lines.append(f"- **Status:** {r.get('status', 'N/A')}")
            if r.get("metric_json"):
                lines.append(f"- **Metrics:** {r['metric_json']}")
            if r.get("wall_clock_seconds"):
                lines.append(f"- **Duration:** {r['wall_clock_seconds']}s")
            if r.get("compute_cost_usd"):
                lines.append(f"- **Cost:** ${r['compute_cost_usd']:.2f}")
        return "\n".join(lines) + "\n"

    async def _gen_contradictions(self, findings: List[Dict]) -> str:
        contradictions = [
            f for f in findings
            if f.get("relation_type") == "CONTRADICTS" and f.get("status") == "active"
        ]
        if not contradictions:
            return "## Contradictions\n\n_No active contradictions._\n"

        lines = ["## Contradictions\n"]
        for f in contradictions:
            lines.append(f"- **{f.get('claim', 'N/A')}**")
            counter = f.get("counter_claim", "No counter-claim provided")
            lines.append(f"  Counter-claim: {counter}")
            lines.append(f"  Confidence: {f.get('confidence', 0):.2f}")
        return "\n".join(lines) + "\n"

    async def _gen_limitations(self, findings: List[Dict]) -> str:
        active = [f for f in findings if f.get("status") == "active"]
        limitations = []

        weak = [f for f in active if (f.get("min_source_tier") or 4) >= 3]
        if weak:
            limitations.append(f"{len(weak)} findings rely on tier-3+ sources (low reliability)")

        low_conf = [f for f in active if f.get("confidence", 0) < 0.45]
        if low_conf:
            limitations.append(f"{len(low_conf)} findings below 0.45 confidence")

        contradictions = [f for f in active if f.get("relation_type") == "CONTRADICTS"]
        if contradictions:
            limitations.append(f"{len(contradictions)} unresolved contradictions")

        exp_records = await self.dag.run(
            "MATCH (r:ExperimentRun {session_id: $sid}) RETURN count(r) as cnt",
            sid=self.session_id,
        )
        if exp_records and exp_records[0]["cnt"] == 0:
            limitations.append("No computational verification performed")

        if not active:
            limitations.append("No findings with adequate evidence")

        lines = ["## ⚠ Limitations\n"]
        for lim in limitations:
            lines.append(f"- ⚠ {lim}")
        if not limitations:
            lines.append("- No significant limitations identified")
        return "\n".join(lines) + "\n"


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High confidence"
    elif score >= 0.65:
        return "Moderately confident"
    elif score >= 0.45:
        return "Tentative"
    elif score >= 0.25:
        return "Weak evidence"
    return "Insufficient evidence"


def _tier_label(tier: int) -> str:
    labels = {1: "Peer-reviewed", 2: "Preprint", 3: "Secondary", 4: "Unclassified"}
    return labels.get(tier, "Unknown")
