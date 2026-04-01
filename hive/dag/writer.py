"""DAG write operations — write_finding(), write_experiment_result(), write_edge()."""

from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hive.dag.client import DAGClient
from hive.schema.finding import Finding
from hive.schema.experiment import ExperimentResult, ExperimentSpec


async def post_finding(driver: "DAGClient", finding: Finding) -> str:
    """
    Write a Finding to the DAG.

    Validates via Pydantic (all rules enforced at schema level).
    Writes to Neo4j.
    Returns finding.id on success.
    """
    await driver.run(
        """
        MERGE (f:Finding {id: $id})
        SET f.session_id = $session_id,
            f.agent_id = $agent_id,
            f.cycle_posted = $cycle_posted,
            f.claim = $claim,
            f.confidence = $confidence,
            f.confidence_rationale = $confidence_rationale,
            f.evidence_type = $evidence_type,
            f.source_urls = $source_urls,
            f.source_tiers = $source_tiers,
            f.min_source_tier = $min_source_tier,
            f.has_numerical_verification = $has_numerical_verification,
            f.cluster_id = $cluster_id,
            f.status = $status,
            f.experiment_run_id = $experiment_run_id,
            f.created_at = $created_at
        """,
        id=finding.id,
        session_id=finding.session_id,
        agent_id=finding.agent_id,
        cycle_posted=finding.cycle_posted,
        claim=finding.claim,
        confidence=finding.confidence,
        confidence_rationale=finding.confidence_rationale,
        evidence_type=finding.evidence_type,
        source_urls=finding.source_urls,
        source_tiers=finding.source_tiers,
        min_source_tier=finding.min_source_tier,
        has_numerical_verification=finding.has_numerical_verification,
        cluster_id=finding.cluster_id,
        status=finding.status,
        experiment_run_id=finding.experiment_run_id,
        created_at=(finding.created_at or datetime.utcnow()).isoformat(),
    )

    # Create edge if relates_to is set
    if finding.relates_to and finding.relation_type:
        edge_query = f"""
        MATCH (a:Finding {{id: $source_id}})
        MATCH (b:Finding {{id: $target_id}})
        MERGE (a)-[r:{finding.relation_type}]->(b)
        SET r.agent_id = $agent_id, r.created_at = $created_at
        """
        if finding.relation_type == "CONTRADICTS":
            edge_query = f"""
            MATCH (a:Finding {{id: $source_id}})
            MATCH (b:Finding {{id: $target_id}})
            MERGE (a)-[r:CONTRADICTS]->(b)
            SET r.agent_id = $agent_id, r.created_at = $created_at,
                r.counter_claim = $counter_claim
            """
            await driver.run(
                edge_query,
                source_id=finding.id,
                target_id=finding.relates_to,
                agent_id=finding.agent_id,
                created_at=datetime.utcnow().isoformat(),
                counter_claim=finding.counter_claim,
            )
        else:
            await driver.run(
                edge_query,
                source_id=finding.id,
                target_id=finding.relates_to,
                agent_id=finding.agent_id,
                created_at=datetime.utcnow().isoformat(),
            )

    # Link to experiment run if present
    if finding.experiment_run_id:
        await driver.run(
            """
            MATCH (f:Finding {id: $finding_id})
            MATCH (r:ExperimentRun {id: $run_id})
            MERGE (f)-[:PRODUCED_BY {created_at: $created_at}]->(r)
            """,
            finding_id=finding.id,
            run_id=finding.experiment_run_id,
            created_at=datetime.utcnow().isoformat(),
        )

    return finding.id


async def write_experiment_spec(driver: "DAGClient", spec: ExperimentSpec) -> str:
    """Write an Experiment node to the DAG."""
    await driver.run(
        """
        MERGE (e:Experiment {id: $id})
        SET e.session_id = $session_id,
            e.hypothesis_finding_id = $hypothesis_finding_id,
            e.backend_type = $backend_type,
            e.goal = $goal,
            e.spec_json = $spec_json,
            e.submitted_by = $submitted_by,
            e.submitted_at = $submitted_at,
            e.status = $status
        """,
        id=spec.id,
        session_id=spec.session_id,
        hypothesis_finding_id=spec.hypothesis_finding_id,
        backend_type=spec.backend_type,
        goal=spec.goal,
        spec_json=spec.model_dump_json(),
        submitted_by=spec.submitted_by,
        submitted_at=(spec.submitted_at or datetime.utcnow()).isoformat(),
        status=spec.status,
    )
    return spec.id


async def post_experiment_result(driver: "DAGClient", result: ExperimentResult) -> str:
    """
    Write ExperimentRun node and create DAG edges.

    Called by the Coordinator after a backend returns.
    Creates PRODUCED_BY edge from resulting Finding to ExperimentRun.
    """
    # Write ExperimentRun node
    await driver.run(
        """
        MERGE (r:ExperimentRun {id: $id})
        SET r.spec_id = $spec_id,
            r.session_id = $session_id,
            r.hypothesis_finding_id = $hypothesis_finding_id,
            r.backend_type = $backend_type,
            r.status = $status,
            r.summary = $summary,
            r.metrics = $metrics,
            r.judgment_outcome = $judgment_outcome,
            r.judgment_confidence = $judgment_confidence,
            r.judgment_reason = $judgment_reason,
            r.artifacts = $artifacts,
            r.environment = $environment,
            r.cost = $cost,
            r.completed_at = $completed_at,
            r.wall_clock_seconds = $wall_clock_seconds
        """,
        id=result.id,
        spec_id=result.spec_id,
        session_id=result.session_id,
        hypothesis_finding_id=result.hypothesis_finding_id,
        backend_type=result.backend_type,
        status=result.status,
        summary=result.summary,
        metrics=str(result.metrics),
        judgment_outcome=result.judgment.outcome,
        judgment_confidence=result.judgment.confidence,
        judgment_reason=result.judgment.reason,
        artifacts=result.artifacts,
        environment=str(result.environment),
        cost=str(result.cost),
        completed_at=(result.completed_at or datetime.utcnow()).isoformat(),
        wall_clock_seconds=result.wall_clock_seconds,
    )

    # Link ExperimentRun to hypothesis Finding
    await driver.run(
        """
        MATCH (r:ExperimentRun {id: $run_id})
        MATCH (f:Finding {id: $finding_id})
        MERGE (r)-[:TESTED {created_at: $created_at}]->(f)
        """,
        run_id=result.id,
        finding_id=result.hypothesis_finding_id,
        created_at=datetime.utcnow().isoformat(),
    )

    return result.id


async def write_edge(
    driver: "DAGClient",
    source_id: str,
    target_id: str,
    edge_type: str,
    agent_id: str,
    **edge_props,
) -> None:
    """Write a typed edge between any two nodes."""
    props_str = ", ".join(f"r.{k} = ${k}" for k in edge_props)
    set_clause = f"r.agent_id = $agent_id, r.created_at = $created_at"
    if props_str:
        set_clause += f", {props_str}"

    query = f"""
    MATCH (a {{id: $source_id}})
    MATCH (b {{id: $target_id}})
    MERGE (a)-[r:{edge_type}]->(b)
    SET {set_clause}
    """

    await driver.run(
        query,
        source_id=source_id,
        target_id=target_id,
        agent_id=agent_id,
        created_at=datetime.utcnow().isoformat(),
        **edge_props,
    )
