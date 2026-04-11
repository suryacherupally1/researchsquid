"""
Hindsight server lifecycle management.

Creates and manages a single Hindsight server instance for the institute.
The server runs in embedded mode (no Docker required) using its own
PostgreSQL instance for memory storage.
"""

import logging
from pathlib import Path
from typing import Any

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


class HindsightManager:
    """
    Manages the Hindsight server lifecycle for the institute.

    Usage:
        manager = HindsightManager(config)
        await manager.start()
        client = manager.client
        # ... use client for retain/recall/reflect ...
        await manager.stop()
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or default_settings
        self._server: Any | None = None
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Hindsight server not started. Call start() first.")
        return self._client

    @property
    def is_running(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        """Start the embedded Hindsight server."""
        from hindsight import HindsightServer, HindsightClient

        data_dir = Path(self._config.hindsight_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        llm_model = (
            self._config.hindsight_llm_model
            or self._config.fast_model
            or "gpt-4o-mini"
        )

        # Use the existing Docker Postgres container instead of embedded pg0
        db_url = self._config.database_url.replace("+asyncpg", "")
        
        self._server = HindsightServer(
            db_url=db_url,
            llm_provider=self._config.hindsight_llm_provider,
            llm_model=llm_model,
            llm_api_key=self._config.openai_api_key,
            llm_base_url=self._config.openai_api_base if self._config.openai_api_base != "https://api.openai.com/v1" else None,
            port=self._config.hindsight_port,
        )
        # Start server with a 5-minute timeout to allow for first-time HuggingFace model downloads
        self._server.start(timeout=300)
        self._client = HindsightClient(base_url=self._server.url)
        logger.info(
            "Hindsight server started at %s (model=%s, data=%s)",
            self._server.url, llm_model, data_dir,
        )

    async def stop(self) -> None:
        """Stop the embedded Hindsight server."""
        if self._server:
            self._server.__exit__(None, None, None)
            self._server = None
            self._client = None
            logger.info("Hindsight server stopped")

    async def create_agent_banks(
        self,
        agent_ids: list[str],
        session_id: str,
        subproblems: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Create private + shared memory banks for a research session.

        Called during _spawn_squids in institute_graph.py. Bank creation
        is implicit in Hindsight — the first retain call creates it.

        Returns:
            Mapping of agent_id → private_bank_id
        """
        if not self._client:
            return {}

        bank_mapping: dict[str, str] = {}

        for agent_id in agent_ids:
            private_bank_id = f"squid-{agent_id}-{session_id}"
            bank_mapping[agent_id] = private_bank_id

            # Seed the bank with the agent's subproblem context
            if subproblems and agent_id in subproblems:
                self._client.retain(
                    bank_id=private_bank_id,
                    content=f"Assigned research subproblem: {subproblems[agent_id]}",
                    context="session-initialization",
                )

        return bank_mapping
