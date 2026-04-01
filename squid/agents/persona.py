"""Agent persona — research specialization profiles for Tier-1 agents."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentPersona(BaseModel):
    """
    Research specialization profile for a Tier-1 agent.

    These are strategy/profile fields only.
    They do NOT modify: tool allowlists, backend permissions, safety rules, budget limits.
    """
    id: str
    agent_id: str
    session_id: str
    revision: int = 1

    # Editable strategy fields
    specialty: str = "general"  # e.g. "pharmacology", "methods_skeptic", "cost_analysis"
    skepticism_level: float = Field(ge=0.0, le=1.0, default=0.5)
    preferred_evidence_types: List[str] = ["empirical", "theoretical"]
    contradiction_aggressiveness: float = Field(ge=0.0, le=1.0, default=0.5)
    source_strictness: float = Field(ge=0.0, le=1.0, default=0.7)  # preference for tier-1/2
    experiment_appetite: float = Field(ge=0.0, le=1.0, default=0.5)  # tendency to propose experiments
    reporting_style: str = "concise"  # "concise" | "detailed" | "critical"

    # Model selection — per-persona LLM tier
    model_tier: Literal["fast", "balanced", "powerful"] = "fast"
    # fast = cheapest (Haiku, GPT-4o-mini, Qwen-turbo)
    # balanced = mid (Sonnet, GPT-4o, Qwen-plus)
    # powerful = best (Opus, o1, Qwen-max)

    # System fields (NOT editable)
    created_at: datetime = None
    updated_at: datetime = None
    revision_history: List[Dict[str, Any]] = []


# Model tier mapping — resolved at runtime from env vars
MODEL_TIERS: Dict[str, Dict[str, str]] = {
    "fast": {
        "env_var": "SQUID_LLM_MODEL_FAST",
        "default": "claude-haiku-4-5",
        "description": "Cheapest, fastest. Good for search, fetch, simple reasoning.",
    },
    "balanced": {
        "env_var": "SQUID_LLM_MODEL_BALANCED",
        "default": "claude-sonnet-4-5",
        "description": "Mid-cost. Good for analysis, clustering, synthesis.",
    },
    "powerful": {
        "env_var": "SQUID_LLM_MODEL_POWERFUL",
        "default": "claude-opus-4",
        "description": "Best quality. Good for final report, complex reasoning, paradigm shifts.",
    },
}


def resolve_model(tier: str) -> str:
    """Resolve model tier to actual model name from env or default."""
    import os
    config = MODEL_TIERS.get(tier, MODEL_TIERS["fast"])
    return os.getenv(config["env_var"], config["default"])


# Pre-built persona templates by domain
PERSONA_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "pharmacology": {
        "specialty": "pharmacology",
        "skepticism_level": 0.6,
        "preferred_evidence_types": ["empirical", "computational"],
        "contradiction_aggressiveness": 0.4,
        "source_strictness": 0.9,
        "experiment_appetite": 0.7,
        "reporting_style": "detailed",
        "model_tier": "balanced",
    },
    "medicinal_chemistry": {
        "specialty": "medicinal_chemistry",
        "skepticism_level": 0.5,
        "preferred_evidence_types": ["empirical", "computational"],
        "contradiction_aggressiveness": 0.5,
        "source_strictness": 0.8,
        "experiment_appetite": 0.8,
        "reporting_style": "detailed",
        "model_tier": "balanced",
    },
    "methods_skeptic": {
        "specialty": "methods_skeptic",
        "skepticism_level": 0.9,
        "preferred_evidence_types": ["empirical"],
        "contradiction_aggressiveness": 0.8,
        "source_strictness": 0.95,
        "experiment_appetite": 0.3,
        "reporting_style": "critical",
        "model_tier": "balanced",
    },
    "cost_analysis": {
        "specialty": "cost_analysis",
        "skepticism_level": 0.5,
        "preferred_evidence_types": ["empirical", "theoretical"],
        "contradiction_aggressiveness": 0.3,
        "source_strictness": 0.7,
        "experiment_appetite": 0.4,
        "reporting_style": "concise",
        "model_tier": "fast",
    },
    "optimization": {
        "specialty": "optimization",
        "skepticism_level": 0.4,
        "preferred_evidence_types": ["computational", "empirical"],
        "contradiction_aggressiveness": 0.5,
        "source_strictness": 0.6,
        "experiment_appetite": 0.9,
        "reporting_style": "detailed",
        "model_tier": "fast",
    },
    "simulation_reliability": {
        "specialty": "simulation_reliability",
        "skepticism_level": 0.8,
        "preferred_evidence_types": ["computational", "empirical"],
        "contradiction_aggressiveness": 0.6,
        "source_strictness": 0.7,
        "experiment_appetite": 0.7,
        "reporting_style": "critical",
        "model_tier": "balanced",
    },
    "literature_critic": {
        "specialty": "literature_critic",
        "skepticism_level": 0.7,
        "preferred_evidence_types": ["theoretical", "empirical"],
        "contradiction_aggressiveness": 0.7,
        "source_strictness": 0.95,
        "experiment_appetite": 0.2,
        "reporting_style": "detailed",
        "model_tier": "fast",
    },
    "devil_advocate": {
        "specialty": "devil_advocate",
        "skepticism_level": 1.0,
        "preferred_evidence_types": ["empirical"],
        "contradiction_aggressiveness": 1.0,
        "source_strictness": 0.9,
        "experiment_appetite": 0.5,
        "reporting_style": "critical",
        "model_tier": "balanced",
    },
    "synthesizer": {
        "specialty": "synthesizer",
        "skepticism_level": 0.5,
        "preferred_evidence_types": ["empirical", "theoretical", "computational"],
        "contradiction_aggressiveness": 0.3,
        "source_strictness": 0.8,
        "experiment_appetite": 0.3,
        "reporting_style": "detailed",
        "model_tier": "powerful",
    },
}


def create_persona(
    agent_id: str,
    session_id: str,
    template: Optional[str] = None,
) -> AgentPersona:
    """Create a persona from a template or defaults."""
    persona_id = f"persona_{uuid.uuid4().hex[:8]}"
    data = PERSONA_TEMPLATES.get(template, {}) if template else {}
    return AgentPersona(
        id=persona_id,
        agent_id=agent_id,
        session_id=session_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        **data,
    )


def generate_persona_prompt(persona: AgentPersona) -> str:
    """Generate a persona-specific addition to the agent system prompt."""
    model_name = resolve_model(persona.model_tier)
    lines = [
        f"## Your Research Persona",
        f"- **Specialty:** {persona.specialty}",
        f"- **Skepticism level:** {persona.skepticism_level:.0%}",
        f"- **Preferred evidence:** {', '.join(persona.preferred_evidence_types)}",
        f"- **Source strictness:** {persona.source_strictness:.0%} (preference for tier-1/2 sources)",
        f"- **Experiment appetite:** {persona.experiment_appetite:.0%} (tendency to propose experiments)",
        f"- **Reporting style:** {persona.reporting_style}",
        f"- **Model:** {model_name} ({persona.model_tier} tier)",
    ]

    if persona.skepticism_level > 0.7:
        lines.append("- You are naturally skeptical. Question assumptions aggressively.")
    if persona.contradiction_aggressiveness > 0.7:
        lines.append("- You actively seek counter-evidence and challenge consensus positions.")
    if persona.source_strictness > 0.8:
        lines.append("- You strongly prefer tier-1 and tier-2 sources. Flag weak sources prominently.")
    if persona.experiment_appetite > 0.7:
        lines.append("- You prefer empirical validation over theoretical reasoning alone.")

    return "\n".join(lines)
