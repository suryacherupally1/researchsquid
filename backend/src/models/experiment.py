"""
Experiment and Result models — sandboxed validation of hypotheses.

An ExperimentSpec describes code to run in a Docker sandbox.
An ExperimentResult captures what happened when it ran.
Together they close the hypothesis → test → finding loop.
"""

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.base import BaseArtifact


ExperimentStatus = Literal["pending", "running", "completed", "failed", "timeout"]


class ExperimentSpec(BaseModel):
    """
    A specification for a sandboxed Python experiment.

    Created by a Squid agent to test a Hypothesis. The spec includes
    the code to run, any data to inject, and what outcome would confirm
    or deny the hypothesis.
    """

    code: str = Field(
        ...,
        description="Python code to execute inside the sandbox.",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Pip packages needed (must be from the pre-installed allow-list).",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Max execution time before the experiment is killed.",
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable data passed to the experiment via stdin.",
    )
    expected_outcome: str = Field(
        default="",
        description="What result would confirm or deny the hypothesis.",
    )


class Experiment(BaseArtifact):
    """
    A sandboxed experiment tied to a hypothesis.

    Wraps an ExperimentSpec with lifecycle tracking. The sandbox runner
    updates status as the experiment progresses.
    """

    hypothesis_id: str = Field(
        ...,
        description="ID of the Hypothesis being tested.",
    )
    spec: ExperimentSpec = Field(
        ...,
        description="The experiment specification.",
    )
    status: ExperimentStatus = Field(  # type: ignore[assignment]
        default="pending",
        description="Current execution status.",
    )
    container_id: str = Field(
        default="",
        description="Docker container ID while running.",
    )

    def neo4j_properties(self) -> dict:
        """Flatten spec into top-level properties for Neo4j storage."""
        props = super().neo4j_properties()
        # Neo4j can't store nested objects — serialize spec as JSON string
        spec_data = props.pop("spec")
        props["spec_code"] = spec_data["code"]
        props["spec_timeout"] = spec_data["timeout_seconds"]
        props["spec_expected_outcome"] = spec_data["expected_outcome"]
        props["spec_requirements"] = spec_data["requirements"]
        props["spec_input_data"] = json.dumps(
            spec_data["input_data"],
            ensure_ascii=False,
            sort_keys=True,
        )
        return props


class ExperimentResult(BaseArtifact):
    """
    The output from a completed experiment.

    Captures stdout, stderr, exit code, and any structured artifacts
    the experiment produced. The agent interprets this into a Finding.
    """

    experiment_id: str = Field(
        ...,
        description="ID of the Experiment that produced this result.",
    )
    stdout: str = Field(
        default="",
        description="Standard output from the experiment.",
    )
    stderr: str = Field(
        default="",
        description="Standard error from the experiment.",
    )
    exit_code: int = Field(
        default=-1,
        description="Process exit code (0 = success).",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output data produced by the experiment.",
    )
    interpretation: str = Field(
        default="",
        description="Agent's interpretation of what the result means.",
    )
    execution_time_seconds: float = Field(
        default=0.0,
        description="How long the experiment took to run.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When execution began.",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When execution finished.",
    )

    def neo4j_properties(self) -> dict:
        """Flatten structured result fields into Neo4j-safe properties."""
        props = super().neo4j_properties()
        props["started_at"] = self.started_at.isoformat()
        props["completed_at"] = self.completed_at.isoformat()
        props["artifacts_json"] = json.dumps(
            self.artifacts,
            ensure_ascii=False,
            sort_keys=True,
        )
        props.pop("artifacts", None)
        return props


class FindingInterpretation(BaseModel):
    """Structured output from experiment result interpretation.

    Produced by the LLM when auto-classifying what an experiment
    result means for the hypothesis it tested. This feeds into
    Finding creation and confidence propagation.
    """

    conclusion_type: Literal["supports", "refutes", "inconclusive", "partial"] = Field(
        ...,
        description="Whether the result supports, refutes, or is inconclusive about the hypothesis.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident you are in this conclusion (0.0 = uncertain, 1.0 = definitive).",
    )
    text: str = Field(
        ...,
        description="Clear explanation of what the experiment result means for the hypothesis.",
    )
