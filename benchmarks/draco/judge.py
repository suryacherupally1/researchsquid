"""
LLM-as-judge for DRACO benchmark evaluation.

Implements the binary MET/UNMET verdict per criterion using the same
LLMClient already wired into ResearchSquid, so judge model is
configurable via settings (JUDGE_MODEL env var or config).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Literal

from src.config import settings
from src.llm.client import LLMClient


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of a research response.
Your task is to decide whether a specific criterion is MET or UNMET.

Rules:
- Answer ONLY with a JSON object: {"verdict": "MET"} or {"verdict": "UNMET"}
- Do not add explanations outside the JSON.
- For positive-weight criteria: MET means the response satisfies the requirement.
- For negative-weight criteria (errors/flaws): MET means the error IS present in the response.
- Be strict and precise. Partial satisfaction counts as UNMET.
"""

JUDGE_USER_TEMPLATE = """\
## Research Query
{query}

## System Response
{response}

## Criterion to Evaluate
{requirement}

Verdict (JSON only):"""


@dataclass
class CriterionResult:
    criterion_id: str
    weight: int
    requirement: str
    verdict: Literal["MET", "UNMET"]
    section_id: str


@dataclass
class SectionScore:
    section_id: str
    raw: float
    positive_max: float
    criteria: list[CriterionResult]

    @property
    def normalized(self) -> float:
        if self.positive_max == 0:
            return 0.0
        return max(0.0, min(1.0, self.raw / self.positive_max))


@dataclass
class TaskScore:
    task_id: str
    domain: str
    problem: str
    response: str
    sections: list[SectionScore]

    @property
    def overall_raw(self) -> float:
        return sum(s.raw for s in self.sections)

    @property
    def overall_positive_max(self) -> float:
        return sum(s.positive_max for s in self.sections)

    @property
    def overall_score(self) -> float:
        if self.overall_positive_max == 0:
            return 0.0
        return max(0.0, min(1.0, self.overall_raw / self.overall_positive_max)) * 100

    def section_score(self, section_id: str) -> float:
        for s in self.sections:
            if s.section_id == section_id:
                return s.normalized * 100
        return 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "overall_score": round(self.overall_score, 2),
            "factual_accuracy": round(self.section_score("factual-accuracy"), 2),
            "breadth_depth": round(self.section_score("breadth-and-depth-of-analysis"), 2),
            "presentation": round(self.section_score("presentation-quality"), 2),
            "citation": round(self.section_score("citation-quality"), 2),
            "sections": [
                {
                    "section_id": s.section_id,
                    "normalized_pct": round(s.normalized * 100, 2),
                    "raw": s.raw,
                    "positive_max": s.positive_max,
                    "criteria": [
                        {
                            "id": c.criterion_id,
                            "weight": c.weight,
                            "verdict": c.verdict,
                            "requirement": c.requirement[:120],
                        }
                        for c in s.criteria
                    ],
                }
                for s in self.sections
            ],
        }


class DracoJudge:
    """Evaluates a single ResearchSquid response against a DRACO rubric."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        concurrency: int = 10,
    ) -> None:
        self._llm = llm or LLMClient(settings)
        # Resolve judge model: JUDGE_MODEL env var → openai_model fallback
        self._judge_model: str | None = settings.judge_model or None
        self._sem = asyncio.Semaphore(concurrency)

    async def judge_criterion(
        self,
        query: str,
        response: str,
        requirement: str,
    ) -> Literal["MET", "UNMET"]:
        """Ask the LLM judge for a binary verdict on one criterion."""
        prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            response=response[:12_000],  # guard against huge responses
            requirement=requirement,
        )
        async with self._sem:
            raw = await self._llm.complete(
                prompt=prompt,
                system=JUDGE_SYSTEM_PROMPT,
                max_tokens=32,
                temperature=0.0,
                model_override=self._judge_model,
            )

        text = raw.strip()
        # Parse {"verdict": "MET"} or {"verdict": "UNMET"}
        match = re.search(r'"verdict"\s*:\s*"(MET|UNMET)"', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()  # type: ignore[return-value]
        # Fallback: look for bare keyword
        if "UNMET" in text.upper():
            return "UNMET"
        if "MET" in text.upper():
            return "MET"
        return "UNMET"  # conservative default

    async def score_task(
        self,
        task_id: str,
        domain: str,
        problem: str,
        rubric: dict | str,
        response: str,
    ) -> TaskScore:
        """Score a full task against its DRACO rubric.

        All criteria across all sections are judged in parallel (bounded by
        self._sem), matching the DRACO spec: one LLM call per criterion.
        """
        if isinstance(rubric, str):
            rubric = json.loads(rubric)

        # Flatten all criteria with their section IDs, preserving order
        flat: list[tuple[str, dict]] = [
            (section["id"], criterion)
            for section in rubric["sections"]
            for criterion in section["criteria"]
        ]

        # Fire all judge calls concurrently
        results: list[CriterionResult] = await asyncio.gather(*[
            self._judge_one(problem, response, criterion, section_id)
            for section_id, criterion in flat
        ])

        # Re-group by section
        by_section: dict[str, list[CriterionResult]] = {}
        for r in results:
            by_section.setdefault(r.section_id, []).append(r)

        section_scores: list[SectionScore] = []
        for section in rubric["sections"]:
            criteria_results = by_section.get(section["id"], [])
            positive_max = sum(c.weight for c in criteria_results if c.weight > 0)
            raw = sum(c.weight if c.verdict == "MET" else 0 for c in criteria_results)
            section_scores.append(SectionScore(
                section_id=section["id"],
                raw=raw,
                positive_max=positive_max,
                criteria=criteria_results,
            ))

        return TaskScore(
            task_id=task_id,
            domain=domain,
            problem=problem,
            response=response,
            sections=section_scores,
        )

    async def _judge_one(
        self,
        problem: str,
        response: str,
        criterion: dict,
        section_id: str,
    ) -> CriterionResult:
        verdict = await self.judge_criterion(problem, response, criterion["requirement"])
        return CriterionResult(
            criterion_id=criterion["id"],
            weight=criterion["weight"],
            requirement=criterion["requirement"],
            verdict=verdict,
            section_id=section_id,
        )
