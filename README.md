# ResearchSquid

Two-tier autonomous research system. Tier 1 proposes. Tier 2 validates. DAG remembers.

## What It Does

You give it a research question. AI agents search the web, read papers, propose experiments, run real computations, argue with each other, and accumulate findings into a permanent knowledge graph.

**This is NOT:** A chatbot, a search engine with a summary, or fake agents.

**This is:** Real agents doing real research, with real computation backends.

## Quick Start

```bash
cp .env.example .env  # fill in ANTHROPIC_API_KEY and BRAVE_API_KEY
docker compose up -d
curl -X POST localhost:8000/research \
  -d '{"question": "cheaper alternative to acetaminophen", "modality": "general"}'
```

## Use as a Library

```bash
pip install research-squid
```

```python
from squid import Finding, ExperimentSpec, Session
from squid.dag import post_finding, post_experiment_result
from squid.coordinator import create_session
```

## Use from Claude Code

```bash
squid-mcp --transport stdio
# Add to claude_code MCP config
```

5 MCP tools: `squid_start_session`, `squid_get_status`, `squid_get_summary`, `squid_submit_experiment`, `squid_stop_session`

## Use from WhatsApp/Telegram

Install OpenClaw, add research-squid skill. Message: "research cheaper alternative to acetaminophen"

## Architecture

8 layers:
1. **User Interface** — OpenClaw SKILL.md
2. **Coordinator** — FastAPI, session management, budget, routing
3. **Tier-1 Agents** — LangGraph, 5-25 parallel, web search + reasoning
4. **Knowledge DAG** — Neo4j, permanent knowledge graph
5. **Execution Backend Router** — Routes ExperimentSpecs to typed backends
6. **Cluster Engine** — Embedding + LLM clustering, devil's advocate
7. **Tool Layer** — web_search, fetch_url, python_exec_sandbox, propose_experiment, post_finding
8. **Output** — Calibrated confidence, honest limitations

## Adding a Backend

1. Create `squid/backends/my_backend/executor.py`
2. Implement `BaseBackend.run_experiment()`
3. Add one line to `squid/backends/registry.py`
4. Done

See [docs/adding-a-backend.md](docs/adding-a-backend.md)

## The Single Design Rule

Tier 1 proposes. Tier 2 validates. DAG remembers.
