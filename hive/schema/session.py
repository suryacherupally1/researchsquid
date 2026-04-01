"""Session, SessionConfig, ResearchModality, SessionWorkflow — session lifecycle objects."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ResearchModality:
    GENERAL = "general"
    LLM_OPTIMIZATION = "llm_optimization"
    DRUG_DISCOVERY = "drug_discovery"
    ENGINEERING_SIMULATION = "engineering_simulation"


class SessionWorkflowStep:
    """Linear workflow steps for UI display."""
    QUESTION = 1           # User provides question
    SETUP = 2              # Context, modality, backend config
    AGENT_SEARCH = 3       # Agents searching, posting findings
    EXPERIMENTS = 4        # Experiment queue + runs
    CLUSTERS = 5           # Clusters, contradictions, debates
    REPORT = 6             # Report generation
    INTERVIEW = 7          # Agent interview / follow-up

    LABELS = {
        0: "Initializing",
        1: "Question / Goal",
        2: "Setup",
        3: "Agent Search",
        4: "Experiments",
        5: "Clusters / Debates",
        6: "Report",
        7: "Interview / Follow-up",
    }


class SessionConfig(BaseModel):
    question: str
    modality: str = ResearchModality.GENERAL
    available_backends: List[str] = ["sandbox_python"]
    backend_config: Dict[str, Any] = {}
    llm_budget_usd: float = 20.0
    compute_budget_usd: float = 20.0
    agent_count: int = 10
    user_id: Optional[str] = None


class Session(BaseModel):
    id: str
    question: str
    modality: str
    status: str = "active"
    created_at: datetime = None

    llm_budget_usd: float
    compute_budget_usd: float
    llm_spent_usd: float = 0.0
    compute_spent_usd: float = 0.0

    agent_count: int = 0
    finding_count: int = 0

    workflow_step: int = 1

    def status_summary(self) -> Dict:
        return {
            "session_id": self.id,
            "status": self.status,
            "workflow_step": self.workflow_step,
            "workflow_label": SessionWorkflowStep.LABELS.get(self.workflow_step, "Unknown"),
            "agents": self.agent_count,
            "findings": self.finding_count,
            "llm_spent": f"${self.llm_spent_usd:.2f} / ${self.llm_budget_usd:.2f}",
            "compute_spent": f"${self.compute_spent_usd:.2f} / ${self.compute_budget_usd:.2f}",
        }
