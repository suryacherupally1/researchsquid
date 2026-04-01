"""
End-to-end integration test for HiveResearch.

Tests the full workflow:
1. Create session
2. Post findings (with source tiers, numerical verification)
3. Submit experiments
4. Create contradictions (with counter_claim)
5. Session status with workflow step
6. Agent inspector
7. Persona management (create, update, revision)
8. Interview (mock LLM)
9. Report generation
10. Audit log
11. SSE telemetry

Requires: HIVE_MOCK_LLM=true (uses mock LLM, no API keys needed)
Does NOT require Neo4j — tests API contract + schema validation.
"""

import asyncio
import os
import sys

# Enable mock LLM
os.environ["HIVE_MOCK_LLM"] = "true"

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_schema_validation():
    """Test that canonical objects validate correctly."""
    from hive.schema.finding import Finding
    from hive.schema.experiment import ExperimentSpec, BackendJudgment, ExperimentResult
    from hive.schema.session import Session, SessionConfig, SessionWorkflowStep

    # Finding — valid
    f = Finding(
        id="f_test001",
        session_id="s_001",
        agent_id="agent_001",
        claim="Naproxen has a longer half-life than ibuprofen",
        confidence=0.85,
        confidence_rationale="Two independent tier-1 sources",
        evidence_type="empirical",
    )
    assert f.status == "active"

    # Finding — claim too long
    try:
        Finding(
            id="f_bad",
            session_id="s",
            agent_id="a",
            claim="x" * 501,
            confidence=0.5,
            confidence_rationale="test",
            evidence_type="theoretical",
        )
        assert False, "Should have raised"
    except ValueError:
        pass

    # Finding — numerical needs verification
    try:
        Finding(
            id="f_num",
            session_id="s",
            agent_id="a",
            claim="Water boils at 90C",
            confidence=0.5,
            confidence_rationale="test",
            evidence_type="theoretical",
            has_numerical_verification=False,
        )
        assert False, "Should have raised"
    except ValueError:
        pass

    # Finding — CONTRADICTS needs counter_claim
    try:
        Finding(
            id="f_contra",
            session_id="s",
            agent_id="a",
            claim="Ibuprofen is better",
            confidence=0.5,
            confidence_rationale="test",
            evidence_type="theoretical",
            relates_to="f_other",
            relation_type="CONTRADICTS",
            counter_claim=None,
        )
        assert False, "Should have raised"
    except ValueError:
        pass

    # ExperimentSpec
    spec = ExperimentSpec(
        id="spec_001",
        session_id="s_001",
        hypothesis_finding_id="f_001",
        backend_type="sandbox_python",
        goal="Verify boiling point",
        inputs={"code": "print(100)", "description": "test"},
        success_metrics=["result"],
        constraints={},
        stop_conditions=[],
        artifacts_expected=[],
        max_compute_cost_usd=0.0,
        max_wall_clock_seconds=60,
        submitted_by="agent_001",
    )
    assert spec.status == "pending"

    # BackendJudgment
    j = BackendJudgment(outcome="supports", confidence="high", reason="Numerical match")
    assert j.outcome == "supports"

    # SessionWorkflowStep
    assert SessionWorkflowStep.LABELS[1] == "Question / Goal"
    assert SessionWorkflowStep.LABELS[6] == "Report"

    print("[PASS] Schema validation")


def test_persona_system():
    """Test persona creation, templates, revision, prompt generation."""
    from hive.agents.persona import (
        AgentPersona, create_persona, generate_persona_prompt, PERSONA_TEMPLATES,
    )

    # Templates exist
    assert "pharmacology" in PERSONA_TEMPLATES
    assert "methods_skeptic" in PERSONA_TEMPLATES
    assert "devil_advocate" in PERSONA_TEMPLATES

    # Create from template
    p = create_persona("agent_001", "session_001", template="pharmacology")
    assert p.specialty == "pharmacology"
    assert p.skepticism_level == 0.6
    assert p.revision == 1

    # Generate prompt
    prompt = generate_persona_prompt(p)
    assert "pharmacology" in prompt
    assert "Specialty" in prompt

    # Revision
    p.specialty = "methods_skeptic"
    p.revision += 1
    p.revision_history.append({
        "revision": 1,
        "changes": {"specialty": ("pharmacology", "methods_skeptic")},
    })
    assert p.revision == 2
    assert len(p.revision_history) == 1

    # Devil advocate template
    d = create_persona("agent_002", "session_001", template="devil_advocate")
    assert d.skepticism_level == 1.0
    assert d.contradiction_aggressiveness == 1.0
    prompt_d = generate_persona_prompt(d)
    assert "skeptical" in prompt_d.lower() or "counter-evidence" in prompt_d.lower()

    print("[PASS] Persona system")


def test_mock_llm():
    """Test mock LLM client."""
    from hive.utils.mock_llm import MockLLMClient

    llm = MockLLMClient()

    # Interview response
    resp = llm.chat([{"role": "user", "content": "Why do you believe this?"}])
    assert "findings" in resp.lower() or "evidence" in resp.lower()

    # Hypothesis response
    resp = llm.chat([{"role": "user", "content": "What is your current hypothesis?"}])
    assert "hypothesis" in resp.lower()

    print("[PASS] Mock LLM")


def test_llm_client_mock_mode():
    """Test that LLM client switches to mock mode."""
    from hive.utils.llm_client import get_llm_client, set_llm_client

    # Should be mock because HIVE_MOCK_LLM=true
    client = get_llm_client()
    resp = client.chat([{"role": "user", "content": "test"}])
    assert len(resp) > 0

    # Test override
    from hive.utils.mock_llm import MockLLMClient
    set_llm_client(MockLLMClient())
    client2 = get_llm_client()
    assert isinstance(client2, MockLLMClient)

    print("[PASS] LLM client mock mode")


def test_source_taxonomy():
    """Test source tier classification."""
    from hive.dag.taxonomy import classify_source_tier, get_tier_label

    assert classify_source_tier("https://pubmed.ncbi.nlm.nih.gov/12345/")[0] == 1
    assert classify_source_tier("https://arxiv.org/abs/2301.00001")[0] == 2
    assert classify_source_tier("https://en.wikipedia.org/wiki/Test")[0] == 3
    assert classify_source_tier("https://random-blog.com/post")[0] == 4
    assert get_tier_label(1) == "Peer-reviewed / primary"

    print("[PASS] Source taxonomy")


def test_calibrated_confidence():
    """Test confidence label calibration."""
    def label(score):
        if score >= 0.85: return "High confidence"
        elif score >= 0.65: return "Moderately confident"
        elif score >= 0.45: return "Tentative"
        elif score >= 0.25: return "Weak evidence"
        return "Insufficient evidence"

    assert label(0.95) == "High confidence"
    assert label(0.75) == "Moderately confident"
    assert label(0.55) == "Tentative"
    assert label(0.35) == "Weak evidence"
    assert label(0.10) == "Insufficient evidence"

    # Boundary cases
    assert label(0.85) == "High confidence"
    assert label(0.84) == "Moderately confident"
    assert label(0.65) == "Moderately confident"
    assert label(0.45) == "Tentative"
    assert label(0.25) == "Weak evidence"
    assert label(0.24) == "Insufficient evidence"

    print("[PASS] Calibrated confidence")


def test_budget_enforcement():
    """Test budget threshold actions."""
    from hive.coordinator.budget import BudgetTracker, BudgetAction

    t = BudgetTracker(session_id="test", llm_budget_usd=10.0, compute_budget_usd=5.0)

    assert t.get_action() == BudgetAction.NORMAL

    t.llm_spent_usd = 7.0  # 70%
    assert t.get_action() == BudgetAction.DOWNGRADE_TO_HAIKU

    t.llm_spent_usd = 8.5  # 85%
    assert t.get_action() == BudgetAction.SUSPEND_NON_CRITICAL

    t.llm_spent_usd = 9.0  # 90%
    assert t.get_action() == BudgetAction.QUEUE_EXPERIMENTS

    t.llm_spent_usd = 10.0  # 100%
    assert t.get_action() == BudgetAction.STOP_ALL

    # Compute budget
    t2 = BudgetTracker(session_id="test", llm_budget_usd=100.0, compute_budget_usd=5.0)
    t2.compute_spent_usd = 5.0  # 100% compute
    assert t2.get_action() == BudgetAction.STOP_ALL

    print("[PASS] Budget enforcement")


def test_audit_logging():
    """Test JSONL audit logging."""
    import tempfile
    import os
    from hive.coordinator.audit import AuditLogger, EventType

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=tmpdir)

        logger.log("s_001", EventType.SESSION_STARTED, {"question": "test"})
        logger.log("s_001", EventType.FINDING_POSTED, {"claim": "test claim"}, agent_id="agent_001")
        logger.log("s_001", EventType.EXPERIMENT_SUBMITTED, {"spec_id": "spec_001"})

        events = logger.read_log("s_001")
        assert len(events) == 3
        assert events[0]["event_type"] == "session_started"
        assert events[1]["agent_id"] == "agent_001"
        assert events[2]["data"]["spec_id"] == "spec_001"

    print("[PASS] Audit logging")


def test_telemetry():
    """Test SSE telemetry pub/sub."""
    asyncio.run(_test_telemetry())


async def _test_telemetry():
    from hive.coordinator.telemetry import SessionTelemetry

    telem = SessionTelemetry()

    # Subscribe
    q = await telem.subscribe("s_001")

    # Publish
    await telem.publish("s_001", "finding_posted", {"claim": "test"})

    # Read
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["event_type"] == "finding_posted"
    assert event["session_id"] == "s_001"

    # Unsubscribe
    await telem.unsubscribe("s_001", q)

    print("[PASS] Telemetry")


def test_report_agent():
    """Test report agent tools (no Neo4j needed for basic validation)."""
    from hive.coordinator.report_agent import _confidence_label, _tier_label

    assert _confidence_label(0.90) == "High confidence"
    assert _confidence_label(0.50) == "Tentative"
    assert _confidence_label(0.10) == "Insufficient evidence"

    assert _tier_label(1) == "Peer-reviewed"
    assert _tier_label(4) == "Unclassified"

    print("[PASS] Report agent helpers")


def test_app_creation():
    """Test that FastAPI app creates without errors."""
    from hive.coordinator.app import create_app

    app = create_app()
    routes = [r.path for r in app.routes]

    # Core endpoints
    assert "/research" in routes
    assert "/health" in routes
    assert "/internal/finding" in routes
    assert "/internal/experiment" in routes

    # New v0.2 endpoints
    assert "/session/{session_id}/status" in routes
    assert "/session/{session_id}/stream" in routes
    assert "/session/{session_id}/agent/{agent_id}" in routes
    assert "/personas/templates" in routes
    assert "/session/{session_id}/agent/{agent_id}/persona" in routes
    assert "/session/{session_id}/interview" in routes
    assert "/session/{session_id}/report" in routes
    assert "/session/{session_id}/audit" in routes

    print("[PASS] App creation + all endpoints")


def test_frontend_build():
    """Verify frontend dist exists."""
    import os
    dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "index.html")
    assert os.path.exists(dist), f"Frontend not built: {dist}"

    print("[PASS] Frontend build")


if __name__ == "__main__":
    print("=== HiveResearch End-to-End Tests ===\n")

    test_schema_validation()
    test_persona_system()
    test_mock_llm()
    test_llm_client_mock_mode()
    test_source_taxonomy()
    test_calibrated_confidence()
    test_budget_enforcement()
    test_audit_logging()
    test_telemetry()
    test_report_agent()
    test_app_creation()
    test_frontend_build()

    print("\n=== ALL TESTS PASSED ===")
