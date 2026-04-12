"""
Central configuration for the Research Institute.

All settings are loaded from environment variables (or the repo-root .env file)
using Pydantic Settings. Every component receives its config via dependency
injection - nothing reads os.environ directly.

This is the single source of truth for tuneable values.
"""

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings, populated from environment variables.

    NOTE: Settings is used as both config source and DI container.
    New code should accept Settings as a constructor parameter, not
    import the singleton. See: docs/architecture/inconsistencies.md #18
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"),
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_API_BASE", "LLM_BASE_URL"),
    )
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("OPENAI_MODEL", "LLM_MODEL_NAME"),
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Per-tier model selection
    fast_model: str = Field(
        default="",
        validation_alias=AliasChoices("FAST_MODEL", "SQUID_LLM_MODEL_FAST"),
    )
    balanced_model: str = Field(
        default="",
        validation_alias=AliasChoices("BALANCED_MODEL", "SQUID_LLM_MODEL_BALANCED"),
    )
    powerful_model: str = Field(
        default="",
        validation_alias=AliasChoices("POWERFUL_MODEL", "SQUID_LLM_MODEL_POWERFUL"),
    )
    judge_model: str = Field(
        default="",
        description="Model for LLM-as-judge evaluation. Empty = use openai_model.",
        validation_alias=AliasChoices("JUDGE_MODEL"),
    )

    # LLM temperatures
    temperature_director: float = 0.5
    temperature_archetype_design: float = 0.7
    temperature_squid: float = 0.5
    temperature_reviewer: float = 0.5
    temperature_controller: float = 0.3
    temperature_adjudicator: float = 0.3
    temperature_counter_response: float = 0.5
    temperature_briefing: float = 0.3
    temperature_synthesizer: float = 0.5
    temperature_default_text: float = 0.7
    temperature_default_structured: float = 0.3

    # LLM token and retry limits
    max_tokens_default: int = 4096
    max_tokens_adjudicator: int = 256
    max_tokens_counter_response: int = 512
    max_tokens_briefing: int = 512
    max_tokens_synthesizer: int = 4096
    max_tokens_retry_cap: int = 16384
    max_retries_structured: int = 2
    llm_retry_token_multiplier: int = 2

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "researchsquid"

    # PostgreSQL
    database_url: str = (
        "postgresql+asyncpg://squid:researchsquid@localhost:5432/squid"
    )

    # Search
    tavily_api_key: str = ""
    search_max_results: int = 5
    squid_search_max_results: int = 3

    # Sandbox
    sandbox_image: str = "squid-sandbox:latest"
    # Hard kill timeout for the Docker container (seconds).
    # If the process inside doesn't exit within this time, Docker kills it.
    sandbox_timeout: int = 60
    sandbox_memory_limit: str = "256m"
    sandbox_network: str = "none"
    sandbox_timeout_hard_cap: int = 300
    sandbox_stdout_cap: int = 10000
    sandbox_stderr_cap: int = 5000
    sandbox_cpu_period: int = 100000
    sandbox_cpu_quota: int = 50000
    # Default timeout for ExperimentSpec (seconds).
    # Individual experiments can override this in their spec.
    # Must be <= sandbox_timeout to ensure Docker kills the container
    # before the experiment timeout fires.
    default_experiment_timeout_seconds: int = 60

    # Research defaults
    default_agents: int = 3
    default_budget_usd: float = 10.0
    default_iterations: int = 5
    budget_per_agent_target: float = 2.0  # Target USD per agent
    min_agent_budget_usd: float = 1.0     # Minimum USD per agent
    session_id_length: int = 12
    data_dir: str = str(REPO_ROOT / "data" / "sources")

    # Agent lifecycle
    agent_pause_empty_threshold: int = 2
    agent_pause_score_threshold: float = 0.15
    min_agent_budget: int = 2

    # Archetype and persona limits
    max_archetypes: int = 20
    min_archetypes: int = 3
    archetype_trait_variance: float = 0.1

    # Persona defaults
    persona_default_specialty: str = "general"
    persona_default_preferred_evidence_types: list[str] = Field(
        default_factory=lambda: ["empirical", "theoretical"]
    )
    persona_default_skepticism_level: float = 0.5
    persona_default_contradiction_aggressiveness: float = 0.5
    persona_default_source_strictness: float = 0.7
    persona_default_experiment_appetite: float = 0.5
    persona_default_reporting_style: str = "concise"
    persona_default_motivation: str = "truth_seeking"
    persona_default_collaboration_style: str = "collaborative"
    persona_default_risk_tolerance: float = 0.5
    persona_default_novelty_bias: float = 0.5
    persona_default_model_tier: str = "fast"
    persona_high_threshold: float = 0.7
    persona_low_threshold: float = 0.3
    persona_source_strictness_threshold: float = 0.8

    # Clustering and debate
    recluster_interval: int = 2
    min_clusters: int = 2
    max_clusters: int = 15
    cluster_stance_threshold: float = 0.1
    max_debate_pairs_per_cluster: int = 3
    max_intra_cluster_peers: int = 3
    max_hypotheses_per_peer: int = 2
    max_adjudications_per_round: int = 10

    # Similarity and deduplication
    hypothesis_dedup_threshold: float = 0.85
    dedup_search_top_k: int = 5

    # Retrieval limits
    retrieval_default_top_k: int = 15
    retrieval_source_chunks_top_k: int = 10
    retrieval_hypotheses_top_k: int = 10
    retrieval_notes_top_k: int = 10
    retrieval_agent_context_top_k: int = 20

    # Graph query limits
    provenance_max_depth: int = 10
    neighbor_limit: int = 50
    graph_findings_limit: int = 100
    graph_pending_experiments_limit: int = 20
    graph_findings_synthesis_limit: int = 50
    graph_experiment_results_limit: int = 20
    graph_contradictions_prompt_limit: int = 10

    # Artifact defaults
    review_default_confidence: float = 0.5
    reviewer_refutation_confidence: float = 0.7
    note_default_confidence: float = 0.7
    hypothesis_default_confidence: float = 0.5
    relation_default_weight: float = 0.5
    dedup_relation_weight: float = 0.8

    # Reputation scoring weights
    reputation_baseline: float = 0.5
    reputation_hypothesis_weight: float = 0.4
    reputation_upheld_bonus: float = 0.05
    reputation_experiment_weight: float = 0.2
    reputation_productivity_cap: float = 0.15
    reputation_productivity_per_item: float = 0.01
    reputation_empty_penalty: float = 0.1

    # Belief vector weights
    belief_authorship_weight: float = 1.0
    belief_supports_weight: float = 0.8
    belief_extends_weight: float = 0.5
    belief_questions_weight: float = -0.3
    belief_contradicts_weight: float = -0.7
    belief_refutes_weight: float = -1.0
    belief_depends_on_weight: float = 0.3
    belief_derived_from_weight: float = 0.4
    belief_finding_supports_weight: float = 0.7
    belief_finding_refutes_weight: float = -0.9
    belief_finding_inconclusive_weight: float = 0.0
    belief_finding_partial_weight: float = 0.3

    # Event bus
    event_bus_max_history: int = 1000

    # Iteration briefing
    briefing_top_hypotheses_limit: int = 10
    briefing_contradictions_limit: int = 5

    # Shared fallback archetypes
    fallback_archetypes: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "Methodological Skeptic",
                "description": (
                    "Questions assumptions, methodology, and evidence quality. "
                    "Demands rigorous proof."
                ),
                "base_traits": {
                    "skepticism_level": 0.85,
                    "contradiction_aggressiveness": 0.75,
                    "source_strictness": 0.9,
                    "experiment_appetite": 0.4,
                    "risk_tolerance": 0.2,
                    "novelty_bias": 0.3,
                },
                "suggested_specialties": ["methodology", "epistemology"],
                "suggested_evidence_types": ["empirical", "theoretical"],
                "reporting_style": "critical",
                "motivation": "falsification",
                "collaboration_style": "adversarial",
                "model_tier": "balanced",
            },
            {
                "name": "Empirical Investigator",
                "description": (
                    "Favors data and experiments over theory. "
                    "Proposes concrete tests for every claim."
                ),
                "base_traits": {
                    "skepticism_level": 0.5,
                    "contradiction_aggressiveness": 0.4,
                    "source_strictness": 0.7,
                    "experiment_appetite": 0.9,
                    "risk_tolerance": 0.5,
                    "novelty_bias": 0.4,
                },
                "suggested_specialties": ["experimental_design", "data_analysis"],
                "suggested_evidence_types": ["empirical", "theoretical"],
                "reporting_style": "detailed",
                "motivation": "truth_seeking",
                "collaboration_style": "collaborative",
                "model_tier": "fast",
            },
            {
                "name": "Integrative Generalist",
                "description": (
                    "Connects ideas across domains. Finds patterns others miss "
                    "by bridging specialties."
                ),
                "base_traits": {
                    "skepticism_level": 0.4,
                    "contradiction_aggressiveness": 0.3,
                    "source_strictness": 0.5,
                    "experiment_appetite": 0.5,
                    "risk_tolerance": 0.6,
                    "novelty_bias": 0.7,
                },
                "suggested_specialties": ["interdisciplinary", "synthesis"],
                "suggested_evidence_types": ["empirical", "theoretical"],
                "reporting_style": "concise",
                "motivation": "consensus_building",
                "collaboration_style": "collaborative",
                "model_tier": "balanced",
            },
        ]
    )

    # ── Evidence Loop ────────────────────────────────────────────────
    evidence_support_weight: float = Field(
        default=0.15,
        description="How much a 'supports' finding nudges confidence up.",
    )
    evidence_refute_weight: float = Field(
        default=0.25,
        description="How much a 'refutes' finding nudges confidence down. "
        "Asymmetric — refutation is a stronger signal than support.",
    )
    evidence_partial_weight: float = Field(
        default=0.05,
        description="How much a 'partial' finding nudges confidence up.",
    )

    # ── Parallel Experiments ─────────────────────────────────────────
    max_parallel_experiments: int = Field(
        default=4,
        description="Concurrency cap for sandbox container execution.",
    )

    # ── Convergence Detection ─────────────────────────────────────────
    convergence_threshold: float = Field(
        default=0.75,
        description="Auto-stop when convergence score exceeds this.",
    )

    # ── Hindsight Memory ─────────────────────────────────────────────
    hindsight_enabled: bool = Field(
        default=True,
        description="Enable Hindsight memory layer.",
    )
    hindsight_data_dir: str = Field(
        default=str(REPO_ROOT / "data" / "hindsight"),
        description="Directory for Hindsight server data.",
    )
    hindsight_port: int = Field(
        default=8930,
        description="Port for embedded Hindsight server.",
    )
    hindsight_llm_provider: str = Field(
        default="openai",
        description="LLM provider for Hindsight operations.",
    )
    hindsight_llm_model: str = Field(
        default="",
        description="LLM model for Hindsight. Empty = use fast_model from config.",
    )

    # ── Workspace Layer ─────────────────────────────────────────────────
    workspace_base_path: str = Field(
        default="workspaces",
        description="Base directory for agent workspaces (relative to repo root).",
    )
    workspace_keep_after_session: bool = Field(
        default=True,
        description="If True, workspaces are preserved after session ends (for inspection).",
    )
    workspace_snapshot_on_end: bool = Field(
        default=True,
        description="Zip entire workspace tree at session end for archival.",
    )
    workspace_opencode_timeout: int = Field(
        default=120,
        description="Max seconds for a single OpenCode turn (HTTP call timeout).",
    )
    workspace_opencode_timeout_hard_cap: int = Field(
        default=300,
        description="Hard cap for total OpenCode task time.",
    )
    workspace_opencode_model: str = Field(
        default="",
        description="Model for OpenCode to use. Empty = OpenCode's own default.",
    )
    workspace_max_opencode_tasks_per_session: int = Field(
        default=10,
        description="Max OpenCode tasks (loop invocations) per agent per session.",
    )
    workspace_opencode_max_iterations: int = Field(
        default=3,
        description="Default max Squid→OpenCode review iterations per task.",
    )
    workspace_opencode_review_timeout: int = Field(
        default=30,
        description="Max seconds to wait for each OpenCode turn result.",
    )
    workspace_max_file_size_kb: int = Field(
        default=512,
        description="Max file size in KB for workspace files.",
    )
    workspace_memory_min_entry_length: int = Field(
        default=20,
        description="Minimum character length for a valid memory.md entry.",
    )
    workspace_memory_max_entries: int = Field(
        default=50,
        description="Max entries in memory.md before older entries are archived.",
    )
    workspace_opencode_max_output_size_kb: int = Field(
        default=1024,
        description="Max total workspace size OpenCode can write per task (KB).",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def _resolve_data_dir(cls, value: str | Path) -> str:
        path = Path(value) if value else (REPO_ROOT / "data" / "sources")
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path)

    @field_validator("workspace_base_path", mode="before")
    @classmethod
    def _resolve_workspace_base_path(cls, value: str | Path) -> str:
        path = Path(value) if value else (REPO_ROOT / "workspaces")
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path)


settings = Settings()
