"""DAG read operations — get_context(), get_frontier(), get_clusters()."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hive.dag.client import DAGClient


async def get_context(
    driver: "DAGClient",
    session_id: str,
    agent_id: str,
) -> Dict[str, Any]:
    """
    Agent fetches current research context.

    Returns frontier findings, pending experiments, paradigm shifts.
    """
    frontier = await get_frontier(driver, session_id)
    pending = await get_pending_experiments(driver, session_id)
    shifts = await get_paradigm_shifts(driver)

    return {
        "agent_id": agent_id,
        "session_id": session_id,
        "frontier_findings": frontier,
        "pending_experiments": pending,
        "paradigm_shifts": shifts,
    }


async def get_frontier(driver: "DAGClient", session_id: str) -> List[Dict]:
    """Get active, non-superseded findings — the research frontier."""
    records = await driver.run(
        """
        MATCH (f:Finding {session_id: $session_id, status: 'active'})
        RETURN f
        ORDER BY f.confidence DESC, f.created_at DESC
        LIMIT 50
        """,
        session_id=session_id,
    )
    return [dict(record["f"]) for record in records]


async def get_pending_experiments(driver: "DAGClient", session_id: str) -> List[Dict]:
    """Get pending ExperimentSpecs for a session."""
    records = await driver.run(
        """
        MATCH (e:Experiment {session_id: $session_id, status: 'pending'})
        RETURN e
        ORDER BY e.submitted_at ASC
        """,
        session_id=session_id,
    )
    return [dict(record["e"]) for record in records]


async def get_clusters(driver: "DAGClient", session_id: str) -> List[Dict]:
    """Get clusters for a session."""
    records = await driver.run(
        """
        MATCH (c:Cluster)
        WHERE any(f IN [(f:Finding)-[:BELONGS_TO]->(c) | f] WHERE f.session_id = $session_id)
        RETURN c
        ORDER BY c.total_citation_weight DESC
        """,
        session_id=session_id,
    )
    return [dict(record["c"]) for record in records]


async def get_paradigm_shifts(driver: "DAGClient") -> List[Dict]:
    """Find challengers that outweigh their parents."""
    records = await driver.run(
        """
        MATCH (parent:Finding)<-[:CONTRADICTS]-(challenger:Finding)
        WHERE parent.status = 'active'
        WITH parent, challenger,
             sum([(parent)<-[:SUPPORTS]-(s) | s.weight]) as parent_weight,
             sum([(challenger)<-[:SUPPORTS]-(s) | s.weight]) as challenger_weight
        WHERE challenger_weight > parent_weight * 1.5
        RETURN parent.id as parent_id, challenger.id as challenger_id,
               parent_weight, challenger_weight
        ORDER BY (challenger_weight - parent_weight) DESC
        LIMIT 10
        """
    )
    return [dict(record) for record in records]


async def get_session_summary(driver: "DAGClient", session_id: str) -> str:
    """Get natural-language summary of current research findings."""
    frontier = await get_frontier(driver, session_id)
    if not frontier:
        return "No findings yet — research in early stages."

    lines = [f"**{len(frontier)} active findings:**\n"]
    for f in frontier[:10]:
        conf_label = _confidence_label(f.get("confidence", 0))
        lines.append(f"- [{conf_label}] {f.get('claim', 'N/A')}")
    return "\n".join(lines)


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High confidence"
    elif score >= 0.65:
        return "Moderately confident"
    elif score >= 0.45:
        return "Tentative"
    elif score >= 0.25:
        return "Weak evidence"
    else:
        return "Insufficient evidence"
