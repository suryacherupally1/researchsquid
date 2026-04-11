"""
Convergence metrics — quantitative progress tracking for the controller.

Computes three metrics that together tell the controller whether the
research is converging (ready to stop) or still making progress:

1. confidence_ratio — fraction of hypotheses above high-confidence threshold
2. evidence_coverage — fraction of hypotheses that have at least one Finding
3. contradiction_ratio — fraction of hypotheses involved in contradictions

These feed into a single convergence_score ∈ [0, 1] that the controller
uses (alongside LLM reasoning) to decide when to stop.

Formula:
  convergence_score = (confidence_ratio × 0.4)
                    + (evidence_coverage × 0.4)
                    + ((1 - contradiction_ratio) × 0.2)

The score rewards high-confidence, well-evidenced hypotheses and penalizes
unresolved contradictions. A score > 0.75 (configurable) triggers auto-stop.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.graph.queries import GraphQueries
from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Computed convergence metrics for a research session."""

    total_hypotheses: int
    high_confidence_count: int
    evidence_covered_count: int
    contradicted_count: int
    confidence_ratio: float
    evidence_coverage: float
    contradiction_ratio: float
    convergence_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for injection into controller prompt and events."""
        return {
            "total_hypotheses": self.total_hypotheses,
            "high_confidence_count": self.high_confidence_count,
            "evidence_covered_count": self.evidence_covered_count,
            "contradicted_count": self.contradicted_count,
            "confidence_ratio": round(self.confidence_ratio, 3),
            "evidence_coverage": round(self.evidence_coverage, 3),
            "contradiction_ratio": round(self.contradiction_ratio, 3),
            "convergence_score": round(self.convergence_score, 3),
        }

    def summary_text(self) -> str:
        """Human-readable summary for LLM prompts."""
        return (
            f"Convergence: {self.convergence_score:.2f} "
            f"({self.high_confidence_count}/{self.total_hypotheses} high-conf, "
            f"{self.evidence_covered_count}/{self.total_hypotheses} evidenced, "
            f"{self.contradicted_count} contradicted)"
        )


async def compute_convergence(
    queries: GraphQueries,
    session_id: str,
    config: Settings | None = None,
) -> ConvergenceMetrics:
    """
    Compute convergence metrics from the current knowledge graph state.

    Queries:
    1. All active hypotheses → count those with confidence > 0.7
    2. Hypothesis contexts → count those with at least one Finding
    3. Contradictions → count unique hypotheses involved

    Returns:
        ConvergenceMetrics with all ratios and the composite score.
    """
    cfg = config or default_settings

    # 1. Get all active hypotheses
    hypotheses = await queries.get_all_hypotheses(
        status="active", session_id=session_id
    )
    total = len(hypotheses)
    if total == 0:
        return ConvergenceMetrics(
            total_hypotheses=0,
            high_confidence_count=0,
            evidence_covered_count=0,
            contradicted_count=0,
            confidence_ratio=0.0,
            evidence_coverage=0.0,
            contradiction_ratio=0.0,
            convergence_score=0.0,
        )

    # 2. Count high-confidence hypotheses
    HIGH_CONF_THRESHOLD = 0.7
    high_conf = sum(
        1 for h in hypotheses
        if h.get("confidence", 0.5) >= HIGH_CONF_THRESHOLD
    )

    # 3. Count evidence-covered hypotheses (have at least one Finding)
    evidence_covered = 0
    for h in hypotheses:
        ctx = await queries.get_hypothesis_context(h["id"])
        if ctx.get("findings"):
            evidence_covered += 1

    # 4. Count contradicted hypotheses
    contradictions = await queries.get_session_contradictions(
        session_id=session_id
    )
    contradicted_ids: set[str] = set()
    for c in contradictions:
        contradicted_ids.add(c.get("source_id", ""))
        contradicted_ids.add(c.get("target_id", ""))
    # Only count hypothesis IDs that are in our active set
    active_ids = {h["id"] for h in hypotheses}
    contradicted_count = len(contradicted_ids & active_ids)

    # 5. Compute ratios
    confidence_ratio = high_conf / total
    evidence_coverage = evidence_covered / total
    contradiction_ratio = contradicted_count / total

    # 6. Composite score
    convergence_score = (
        confidence_ratio * 0.4
        + evidence_coverage * 0.4
        + (1.0 - contradiction_ratio) * 0.2
    )

    metrics = ConvergenceMetrics(
        total_hypotheses=total,
        high_confidence_count=high_conf,
        evidence_covered_count=evidence_covered,
        contradicted_count=contradicted_count,
        confidence_ratio=confidence_ratio,
        evidence_coverage=evidence_coverage,
        contradiction_ratio=contradiction_ratio,
        convergence_score=convergence_score,
    )

    logger.info("Convergence metrics: %s", metrics.summary_text())
    return metrics
