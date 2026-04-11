"""
Evidence loop — propagates findings back to hypothesis confidence.

This is the mathematical heart of the closed loop. After each debate cycle,
we traverse all active hypotheses, aggregate their linked Findings, and
compute a new confidence score.

The math is deliberately simple (weighted additive, not full Bayesian):
- supports finding → nudge confidence UP by (finding_confidence × weight)
- refutes finding → nudge confidence DOWN by (finding_confidence × weight)
- inconclusive → no change
- partial → small nudge UP
- Clamped to [0.05, 0.95] to avoid certainty lock-in

Why not full Bayes: LLM-generated likelihoods P(D|H) are unreliable.
A calibrated additive model is more stable and interpretable.
"""

import logging
from typing import Any

from src.config import Settings, settings as default_settings
from src.graph.repository import GraphRepository
from src.graph.queries import GraphQueries
from src.events.bus import EventBus
from src.models.events import Event, EventType

logger = logging.getLogger(__name__)


async def propagate_confidence(
    graph: GraphRepository,
    queries: GraphQueries,
    session_id: str,
    event_bus: EventBus,
    config: Settings | None = None,
) -> dict[str, float]:
    """
    Update hypothesis confidence from linked Findings.

    Called after each debate cycle to close the evidence loop.

    Returns:
        Mapping of hypothesis_id → new_confidence for all updated hypotheses.
    """
    cfg = config or default_settings
    hypotheses = await queries.get_all_hypotheses(
        status="active", session_id=session_id
    )

    updates: dict[str, float] = {}

    for hyp in hypotheses:
        hyp_id = hyp["id"]
        context = await queries.get_hypothesis_context(hyp_id)
        findings = context.get("findings", [])

        if not findings:
            continue

        # Start from base prior (0.5) and calculate absolute confidence
        # This prevents double-counting the same findings in subsequent iterations
        confidence = 0.5
        original = hyp.get("confidence", 0.5)

        for f in findings:
            ctype = f.get("conclusion_type", "inconclusive")
            f_conf = f.get("confidence", 0.5)

            if ctype == "supports":
                confidence += f_conf * cfg.evidence_support_weight
            elif ctype == "refutes":
                confidence -= f_conf * cfg.evidence_refute_weight
            elif ctype == "partial":
                confidence += f_conf * cfg.evidence_partial_weight
            # "inconclusive" → no change

        # Clamp to avoid certainty lock-in
        confidence = max(0.05, min(0.95, confidence))

        # Only update if meaningful change
        if abs(confidence - original) > 0.01:
            await graph.update(hyp_id, {"confidence": round(confidence, 4)})
            updates[hyp_id] = confidence

            await event_bus.publish(Event(
                event_type=EventType.ARTIFACT_UPDATED,
                artifact_id=hyp_id,
                payload={
                    "field": "confidence",
                    "old_value": original,
                    "new_value": confidence,
                    "source": "evidence_propagation",
                    "findings_count": len(findings),
                },
            ))

    return updates
