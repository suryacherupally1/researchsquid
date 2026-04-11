"""
Service layer — clean API between the CLI (or future web UI) and
the research engine.

This is the single entry point for starting research, querying status,
and exporting results. It handles all dependency wiring (connections,
repositories, agents) so callers don't need to know about internals.
"""

from typing import Any
from uuid import uuid4

from src.config import Settings, settings as default_settings
from src.db.connection import PostgresConnection
from src.db.migrations import bootstrap_postgres
from src.events.bus import EventBus
from src.graph.connection import Neo4jConnection
from src.graph.queries import GraphQueries
from src.graph.repository import GraphRepository
from src.graph.schema import bootstrap_schema
from src.llm.client import LLMClient
from src.ingest.summarizer import HierarchicalSummarizer
from src.models.agent_state import InstituteState
from src.orchestration.institute_graph import InstituteGraphBuilder
from src.rag.indexer import RAGIndexer
from src.rag.retriever import RAGRetriever
from src.rag.store import VectorStore
from src.sandbox.runner import SandboxRunner
from src.search.arxiv import ArxivSearch
from src.search.tavily import TavilySearch
from src.session_context import reset_current_session_id, set_current_session_id
from src.workspace.manager import WorkspaceManager
from src.memory.server import HindsightManager


class ResearchService:
    """
    High-level service for running the research institute.

    Manages all infrastructure connections and provides a clean
    interface for the CLI. A future FastAPI web server would call
    these same methods.

    Usage:
        service = ResearchService()
        await service.initialize()
        result = await service.start_research(
            question="What mechanisms drive X?",
            sources=["paper.pdf"],
            num_agents=3,
            budget=500,
            max_iterations=5,
        )
        await service.shutdown()
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or default_settings
        self._neo4j: Neo4jConnection | None = None
        self._postgres: PostgresConnection | None = None
        self._event_bus = EventBus(self._config)
        self._llm: LLMClient | None = None
        self._graph: GraphRepository | None = None
        self._queries: GraphQueries | None = None
        self._vector_store: VectorStore | None = None
        self._retriever: RAGRetriever | None = None
        self._indexer: RAGIndexer | None = None
        self._sandbox: SandboxRunner | None = None
        self._tavily: TavilySearch | None = None
        self._arxiv: ArxivSearch | None = None
        self._workspace_manager: WorkspaceManager | None = None
        self._memory_manager: HindsightManager | None = None
        self._initialized = False

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus to subscribe to research events."""
        return self._event_bus

    @property
    def llm(self) -> LLMClient:
        """Access the configured LLM client."""
        self._ensure_initialized()
        return self._llm  # type: ignore[return-value]

    @property
    def graph(self) -> GraphRepository:
        """Access the graph repository."""
        self._ensure_initialized()
        return self._graph  # type: ignore[return-value]

    @property
    def queries(self) -> GraphQueries:
        """Access graph traversal helpers."""
        self._ensure_initialized()
        return self._queries  # type: ignore[return-value]

    @property
    def postgres(self) -> PostgresConnection:
        """Access the PostgreSQL connection manager."""
        self._ensure_initialized()
        return self._postgres  # type: ignore[return-value]

    @property
    def retriever(self) -> RAGRetriever:
        """Access the semantic retriever."""
        self._ensure_initialized()
        return self._retriever  # type: ignore[return-value]

    async def initialize(self) -> None:
        """
        Initialize all connections and bootstrap schemas.

        Must be called before any other method. Idempotent.
        """
        if self._initialized:
            return

        # Initialize connections
        self._neo4j = Neo4jConnection(self._config)
        await self._neo4j.connect()

        self._postgres = PostgresConnection(self._config)
        await self._postgres.connect()

        # Bootstrap schemas
        await bootstrap_schema(self._neo4j)
        await bootstrap_postgres(self._postgres)

        # Initialize components
        self._llm = LLMClient(self._config)

        self._graph = GraphRepository(self._neo4j, self._event_bus)
        self._queries = GraphQueries(self._neo4j)

        self._vector_store = VectorStore(self._postgres, self._llm, self._config)
        self._retriever = RAGRetriever(self._vector_store, self._graph)

        summarizer = HierarchicalSummarizer(self._llm)
        self._indexer = RAGIndexer(
            self._graph, self._vector_store, summarizer, self._event_bus
        )

        self._sandbox = SandboxRunner(self._config, self._event_bus)

        # Optional search providers
        if self._config.tavily_api_key:
            self._tavily = TavilySearch(self._config, self._event_bus)
        self._arxiv = ArxivSearch(self._config, self._event_bus)

        # Workspace layer (optional — disabled if workspace_base_path not set)
        if self._config.workspace_base_path:
            self._workspace_manager = WorkspaceManager(self._config, self._event_bus)
            await self._workspace_manager.initialize()

        # Hindsight memory layer
        if getattr(self._config, "hindsight_enabled", False):
            self._memory_manager = HindsightManager(self._config)
            await self._memory_manager.start()

        self._initialized = True

    async def shutdown(self) -> None:
        """Close all connections and release resources."""
        if self._neo4j:
            await self._neo4j.close()
        if self._postgres:
            await self._postgres.close()
        if self._memory_manager:
            await self._memory_manager.stop()
        self._initialized = False

    async def start_research(
        self,
        question: str,
        sources: list[str] | None = None,
        num_agents: int | None = None,
        budget_usd: float | None = None,
        max_iterations: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Start a full research session.

        Args:
            question: The research question to investigate.
            sources: Optional list of source file paths or URLs.
            num_agents: Number of squid agents (overrides default).
            budget: LLM call budget (overrides default).
            max_iterations: Max research iterations (overrides default).

        Returns:
            The final state dict including the synthesis report.
        """
        self._ensure_initialized()

        resolved_session_id = session_id or uuid4().hex[: self._config.session_id_length]
        token = set_current_session_id(resolved_session_id)

        try:
            # Ingest user-provided sources
            source_ids: list[str] = []
            if sources:
                for source_path in sources:
                    try:
                        if source_path.startswith(("http://", "https://")):
                            sid = await self._indexer.ingest_url(source_path, "system")
                        elif source_path.endswith(".pdf"):
                            sid = await self._indexer.ingest_pdf(source_path, "system")
                        else:
                            sid = await self._indexer.ingest_file(source_path, "system")
                        source_ids.append(sid)
                    except Exception as e:
                        from src.models.events import Event, EventType
                        await self._event_bus.publish(Event(
                            event_type=EventType.ERROR,
                            payload={"error": f"Failed to ingest {source_path}: {e}"},
                        ))

            # Build and run the institute graph
            builder = InstituteGraphBuilder(
                llm=self._llm,
                graph=self._graph,
                queries=self._queries,
                retriever=self._retriever,
                indexer=self._indexer,
                event_bus=self._event_bus,
                sandbox=self._sandbox,
                tavily=self._tavily,
                arxiv_search=self._arxiv,
                config=self._config,
                workspace_manager=self._workspace_manager,
                memory_manager=self._memory_manager,
            )

            compiled = builder.build()

            initial_state: InstituteState = {
                "research_question": question,
                "session_id": resolved_session_id,
                "num_agents": num_agents or self._config.default_agents,
                "budget_total_usd": budget_usd or self._config.default_budget_usd,
                "budget_remaining_usd": budget_usd or self._config.default_budget_usd,
                "max_iterations": max_iterations or self._config.default_iterations,
                "source_ids": source_ids,
                "iteration": 0,
                "should_stop": False,
            }

            result = await compiled.ainvoke(initial_state)
            result["session_id"] = resolved_session_id
            return result
        finally:
            # Clean up lingering workspace servers
            await self._workspace_manager.snapshot_session(resolved_session_id)
            await self._workspace_manager.stop_all_servers(resolved_session_id)
            reset_current_session_id(token)

    async def get_graph_export(self) -> dict[str, Any]:
        """Export the full knowledge graph for visualization."""
        self._ensure_initialized()
        return await self._queries.export_graph()

    async def get_hypotheses(
        self,
        status: str = "active",
        session_id: str | None = None,
    ) -> list[dict]:
        """Get all hypotheses with optional status filter."""
        self._ensure_initialized()
        return await self._queries.get_all_hypotheses(
            status=status,
            session_id=session_id,
        )

    async def get_coverage_stats(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Get coverage statistics across the knowledge graph."""
        self._ensure_initialized()
        return await self._queries.get_coverage_stats(session_id=session_id)

    def _ensure_initialized(self) -> None:
        """Raise if initialize() hasn't been called."""
        if not self._initialized:
            raise RuntimeError(
                "Service not initialized — call initialize() first."
            )

    # ── Workspace inspection methods ──────────────────────────────────────

    async def list_workspaces(self, session_id: str) -> list[dict[str, Any]]:
        """List all workspaces for a session."""
        if not self._workspace_manager:
            return []
        
        import aiofiles
        from pathlib import Path
        
        base = Path(self._config.workspace_base_path) / session_id
        if not base.exists():
            return []
        
        workspaces = []
        for agent_dir in base.iterdir():
            if not agent_dir.is_dir():
                continue
            
            # Calculate total size and file count
            total_size = 0
            file_count = 0
            for f in agent_dir.rglob("*"):
                if f.is_file() and ".history" not in f.parts:
                    total_size += f.stat().st_size
                    file_count += 1
            
            workspaces.append({
                "agent_id": agent_dir.name,
                "path": str(agent_dir),
                "file_count": file_count,
                "size_kb": total_size / 1024,
            })
        
        return workspaces

    async def read_workspace_file(
        self,
        session_id: str,
        agent_id: str,
        path: str,
    ) -> str:
        """Read a file from an agent's workspace."""
        if not self._workspace_manager:
            raise RuntimeError("Workspace layer is not enabled")
        
        return await self._workspace_manager.read_file(agent_id, session_id, path)

    async def list_workspace_files(
        self,
        session_id: str,
        agent_id: str,
        path: str = "",
    ) -> list[dict[str, Any]]:
        """List files in an agent's workspace directory."""
        if not self._workspace_manager:
            return []
        
        from pathlib import Path
        
        root = self._workspace_manager.workspace_root(agent_id, session_id)
        target = root / path if path else root
        
        if not target.exists():
            return []
        
        files = []
        for f in target.rglob("*"):
            if f.is_file() and ".history" not in f.parts:
                rel = f.relative_to(root)
                files.append({
                    "path": str(rel),
                    "size_kb": f.stat().st_size / 1024,
                })
        
        return sorted(files, key=lambda x: x["path"])

    async def get_opencode_sessions(
        self,
        session_id: str,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """Get OpenCode session history for an agent."""
        if not self._workspace_manager:
            return []
        
        from pathlib import Path
        import json
        
        root = self._workspace_manager.workspace_root(agent_id, session_id)
        sessions_path = root / "logs" / "opencode_sessions.json"
        
        if not sessions_path.exists():
            return []
        
        try:
            content = sessions_path.read_text(encoding="utf-8")
            sessions = json.loads(content)
            return sessions if isinstance(sessions, list) else []
        except (json.JSONDecodeError, OSError):
            return []
