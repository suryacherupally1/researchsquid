"""HiveResearch — two-tier autonomous research system.

Tier 1 proposes. Tier 2 validates. DAG remembers.
"""

from hive.schema.finding import EvidenceNode, Finding
from hive.schema.experiment import BackendJudgment, ExperimentResult, ExperimentSpec
from hive.schema.session import Session, SessionConfig, SessionWorkflowStep

# Persona imports available but not eagerly loaded
# from hive.agents.persona import AgentPersona, PERSONA_TEMPLATES, create_persona

__all__ = [
    "EvidenceNode",
    "Finding",
    "BackendJudgment",
    "ExperimentResult",
    "ExperimentSpec",
    "Session",
    "SessionConfig",
    "SessionWorkflowStep",
]
