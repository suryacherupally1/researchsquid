"""Structured audit logging — JSONL event log for all system events."""

import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_COMPLETED = "session_completed"

    # Agent lifecycle
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    AGENT_HYPOTHESIS_UPDATED = "agent_hypothesis_updated"

    # Findings
    FINDING_POSTED = "finding_posted"

    # Contradictions
    CONTRADICTION_OPENED = "contradiction_opened"

    # Experiments
    EXPERIMENT_SUBMITTED = "experiment_submitted"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"

    # Clusters
    CLUSTER_MEMBERSHIP_CHANGED = "cluster_membership_changed"

    # Budget
    BUDGET_UPDATED = "budget_updated"

    # Report
    REPORT_PROGRESS = "report_progress"

    # Persona
    PERSONA_REVISION_APPLIED = "persona_revision_applied"

    # Interview
    INTERVIEW_STARTED = "interview_started"
    INTERVIEW_COMPLETED = "interview_completed"


class AuditLogger:
    """Append-only JSONL audit logger."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _get_log_path(self, session_id: str) -> str:
        return os.path.join(self.log_dir, f"{session_id}.jsonl")

    def log(
        self,
        session_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> None:
        """Append an event to the session's JSONL log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "event_type": event_type.value,
            "agent_id": agent_id,
            "data": data,
        }
        path = self._get_log_path(session_id)
        with open(path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_log(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list:
        """Read events from a session's log."""
        path = self._get_log_path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            lines = f.readlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        return entries[offset : offset + limit]


# Singleton
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: str = "logs") -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir)
    return _audit_logger
