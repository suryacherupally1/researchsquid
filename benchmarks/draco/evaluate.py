"""
DRACO benchmark evaluation harness for ResearchSquid.

Usage:
    # Full run (all 100 tasks):
    python benchmarks/draco/evaluate.py

    # Single task by index:
    python benchmarks/draco/evaluate.py --task 0

    # Subset by domain:
    python benchmarks/draco/evaluate.py --domain Finance

    # Dry-run (generate responses only, no judging):
    python benchmarks/draco/evaluate.py --dry-run

    # Load pre-generated responses and judge them:
    python benchmarks/draco/evaluate.py --responses-file results/responses.jsonl --judge-only

Results are written to benchmarks/draco/results/<timestamp>/:
    responses.jsonl   — raw system responses per task
    scores.jsonl      — per-task scores (all 4 axes)
    summary.json      — aggregate stats + per-domain breakdown
    report.md         — human-readable markdown report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure backend/src is on path when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO_ROOT))

from src.api.service import ResearchService
from src.config import settings

from benchmarks.draco.judge import DracoJudge
from benchmarks.draco.reporting import build_summary, write_markdown_report

logger = logging.getLogger(__name__)

DRACO_DATA = Path(__file__).parent / "data" / "draco_test.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Research config defaults ──────────────────────────────────────────────────
DEFAULT_AGENTS = 3
DEFAULT_BUDGET = 5.0       # USD per task — keep low for benchmarking
DEFAULT_ITERATIONS = 3
JUDGE_CONCURRENCY = 8      # parallel LLM judge calls per task


def load_tasks(
    domain: str | None = None,
    task_index: int | None = None,
) -> list[dict]:
    with open(DRACO_DATA) as f:
        tasks = json.load(f)

    if task_index is not None:
        return [tasks[task_index]]
    if domain:
        tasks = [t for t in tasks if t["domain"].lower() == domain.lower()]
    return tasks


_REQUIREMENT_VERBS = re.compile(
    r"^(States?|Explains?|Names?|Cites?|Notes?|Specif(?:ies|y)|Identif(?:ies|y)|"
    r"Compares?|Provides?|Uses?|References?|Quotes?|Applies?|Demonstrates?|"
    r"Describes?|Includes?|Lists?|Calculates?|Shows?|Mentions?|Acknowledges?|"
    r"Addresses?|Asserts?|Confirms?|Defines?|Details?|Discusses?|Evaluates?|"
    r"Highlights?|Indicates?|Outlines?|Presents?|Reports?|Summarizes?|"
    r"Establishes?)\s+",
    re.IGNORECASE,
)


def _to_assertion(requirement: str) -> str:
    """
    Convert a rubric requirement into a declarative assertion.

    'States TWFE coefficient is X'  → 'TWFE coefficient is X'
    'Cites Malinowski (1922)'       → 'Malinowski (1922) is cited here'
    'Names R package did'           → 'R package did is referenced here'
    """
    stripped = _REQUIREMENT_VERBS.sub("", requirement).strip()
    # Capitalise first letter
    if stripped:
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def build_mock_response(task: dict) -> str:
    """
    Build a synthetic 'perfect' response for a DRACO task.

    Each positive-weight criterion is converted from a requirement
    description ("States X") into a direct declarative assertion ("X"),
    so the judge reads it as a response that actually contains the
    information rather than one that merely describes what should be said.

    Negative-weight criteria (errors/penalties) are left completely
    unmentioned. A correct judge must score this at 100%.
    """
    rubric = task["answer"]
    lines = [
        f"Research Report",
        f"Query: {task['problem'][:200]}",
        "",
        "The following comprehensive analysis addresses all aspects of the research question.",
        "",
    ]
    for section in rubric["sections"]:
        lines.append(f"## {section.get('title', section['id'])}")
        lines.append("")
        for c in section["criteria"]:
            if c["weight"] > 0:
                lines.append(f"- {_to_assertion(c['requirement'])}")
        lines.append("")
    return "\n".join(lines)


def extract_final_report(result: dict) -> str:
    """Pull the final synthesis report text out of the service result.

    institute_graph.py returns {"final_report": str, "events": [{"type": "final_report", ...}]}
    """
    if "final_report" in result and result["final_report"]:
        return result["final_report"]

    for event in result.get("events", []):
        if isinstance(event, dict) and event.get("type") == "final_report":
            return event.get("content", "")

    return json.dumps(result, indent=2, default=str)[:8000]


async def run_one_task(
    service: ResearchService | None,
    task: dict,
    dry_run: bool = False,
    mock: bool = False,
) -> dict:
    """Run ResearchSquid on a single DRACO task, return response record."""
    problem = task["problem"]
    t0 = time.time()

    if mock:
        response_text = build_mock_response(task)
    elif dry_run:
        response_text = f"[DRY RUN — no actual research performed for task {task['id']}]"
    else:
        result = await service.start_research(
            question=problem,
            num_agents=DEFAULT_AGENTS,
            budget_usd=DEFAULT_BUDGET,
            max_iterations=DEFAULT_ITERATIONS,
        )
        response_text = extract_final_report(result)

    elapsed = time.time() - t0
    return {
        "id": task["id"],
        "domain": task["domain"],
        "problem": problem,
        "response": response_text,
        "elapsed_s": round(elapsed, 1),
    }


async def judge_responses(
    response_records: list[dict],
    tasks_by_id: dict[str, dict],
    dry_run: bool = False,
) -> list[dict]:
    """Score all response records against DRACO rubrics."""
    judge = DracoJudge(concurrency=JUDGE_CONCURRENCY)
    score_records = []

    for rec in response_records:
        task = tasks_by_id[rec["id"]]
        if dry_run:
            print(f"  [dry-run] would judge task {rec['id']} ({rec['domain']})")
            continue

        print(f"  Judging {rec['id']} ({rec['domain']})...")
        score = await judge.score_task(
            task_id=task["id"],
            domain=task["domain"],
            problem=task["problem"],
            rubric=task["answer"],
            response=rec["response"],
        )
        score_dict = score.to_dict()
        score_dict["elapsed_s"] = rec.get("elapsed_s", 0)
        score_records.append(score_dict)
        print(
            f"    → overall={score_dict['overall_score']:.1f}%  "
            f"factual={score_dict['factual_accuracy']:.1f}%  "
            f"breadth={score_dict['breadth_depth']:.1f}%  "
            f"presentation={score_dict['presentation']:.1f}%  "
            f"citation={score_dict['citation']:.1f}%"
        )

    return score_records


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tasks = load_tasks(domain=args.domain, task_index=args.task)
    if not tasks:
        print(f"No tasks found (domain={args.domain!r}, task={args.task}). Exiting.")
        return

    tasks_by_id = {t["id"]: t for t in tasks}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_model = settings.judge_model or settings.openai_model
    print(f"\nDRACO Benchmark — ResearchSquid")
    print(f"Tasks: {len(tasks)}  |  Output: {out_dir}")
    print(f"Research model: {settings.openai_model}  |  Judge model: {judge_model}")
    print(f"Dry run: {args.dry_run}  |  Judge only: {args.judge_only}\n")

    # ── Phase 1: Generate responses ───────────────────────────────────────────
    responses_path = out_dir / "responses.jsonl"

    if args.judge_only:
        if not args.responses_file:
            print("Error: --judge-only requires --responses-file. Exiting.")
            return
        print("Loading pre-generated responses...")
        response_records = []
        with open(args.responses_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec["id"] in tasks_by_id:
                        response_records.append(rec)
    else:
        print("Phase 1: Generating research responses...")
        if args.mock or args.dry_run:
            # No service needed — responses are synthetic
            service = None
        else:
            service = ResearchService()
            await service.initialize()

        response_records = []
        try:
            for i, task in enumerate(tasks, 1):
                print(f"  [{i}/{len(tasks)}] {task['domain']}: {task['problem'][:80]}...")
                rec = await run_one_task(service, task, dry_run=args.dry_run, mock=args.mock)
                response_records.append(rec)
                with open(responses_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        finally:
            if service is not None:
                await service.shutdown()

    print(f"\nResponses saved → {responses_path}")

    # ── Phase 2: Judge responses ──────────────────────────────────────────────
    print("\nPhase 2: Judging responses against DRACO rubrics...")
    score_records = await judge_responses(
        response_records, tasks_by_id, dry_run=args.dry_run
    )

    if not score_records:
        print("No scores generated (dry run or empty). Done.")
        return

    scores_path = out_dir / "scores.jsonl"
    with open(scores_path, "w") as f:
        for rec in score_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Scores saved → {scores_path}")

    # ── Phase 3: Summarize & report ───────────────────────────────────────────
    summary = build_summary(score_records)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    report_path = out_dir / "report.md"
    write_markdown_report(summary, score_records, report_path)
    print(f"Report saved → {report_path}")

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"DRACO Results — {len(score_records)} tasks")
    print(f"{'='*60}")
    print(f"Overall score:       {summary['overall']['mean']:.1f}%")
    print(f"Factual accuracy:    {summary['axes']['factual_accuracy']['mean']:.1f}%")
    print(f"Breadth & depth:     {summary['axes']['breadth_depth']['mean']:.1f}%")
    print(f"Presentation:        {summary['axes']['presentation']['mean']:.1f}%")
    print(f"Citation quality:    {summary['axes']['citation']['mean']:.1f}%")
    print(f"{'='*60}")
    print("\nPer-domain breakdown:")
    for domain, stats in sorted(summary["by_domain"].items()):
        print(f"  {domain:<30} {stats['mean']:.1f}%  (n={stats['n']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DRACO benchmark against ResearchSquid"
    )
    parser.add_argument(
        "--task", type=int, default=None,
        help="Run a single task by 0-based index",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Filter to a specific domain (e.g. Finance, Academic, Medicine)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help=(
            "Use a synthetic 'perfect' response built from the rubric itself. "
            "Every positive criterion is stated verbatim; negative criteria are omitted. "
            "A correct judge must return 100%% — use this to validate the judge model."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual research + judging; useful for testing the pipeline",
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help="Skip research phase; load responses from --responses-file",
    )
    parser.add_argument(
        "--responses-file", type=str, default=None,
        help="Path to existing responses.jsonl (for --judge-only mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
