"""
pgvector store configuration for LlamaIndex.

Sets up the vector store backed by PostgreSQL + pgvector extension.
All embeddings (source chunks, notes, hypotheses, etc.) go here
for semantic retrieval by agents.
"""

import json
from typing import Any

from sqlalchemy import text

from src.config import Settings, settings as default_settings
from src.db.connection import PostgresConnection
from src.llm.client import LLMClient
from src.session_context import get_current_session_id


class VectorStore:
    """
    Manages embedding storage and retrieval via pgvector.

    Provides a simple interface for storing and querying embeddings
    without exposing LlamaIndex internals. This makes it easy to
    swap vector store implementations later.

    Usage:
        store = VectorStore(pg_conn, llm_client, settings)
        await store.store_embedding(artifact_id, "hypothesis", "text...", embedding)
        results = await store.search("antibiotic resistance", top_k=10)
    """

    def __init__(
        self,
        connection: PostgresConnection,
        llm: LLMClient,
        config: Settings | None = None,
    ) -> None:
        self._conn = connection
        self._llm = llm
        self._config = config or default_settings
        self._dimension = self._config.embedding_dimension

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return str(value or "").replace("\x00", "")

    @classmethod
    def _sanitize_jsonish(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._sanitize_text(value)
        if isinstance(value, dict):
            return {
                cls._sanitize_text(str(key)): cls._sanitize_jsonish(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize_jsonish(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_jsonish(item) for item in value]
        if isinstance(value, set):
            return [cls._sanitize_jsonish(item) for item in value]
        return value

    async def store_embedding(
        self,
        artifact_id: str,
        artifact_type: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Store a text embedding for an artifact.

        If no embedding is provided, one is generated via the LLM client.

        Args:
            artifact_id: ID of the artifact being embedded.
            artifact_type: Type label (e.g., "source_chunk", "hypothesis").
            content: The text content to embed.
            embedding: Pre-computed embedding vector, or None to generate.
            metadata: Optional JSON metadata to store alongside.

        Returns:
            The UUID of the embedding row.
        """
        content = self._sanitize_text(content)
        if embedding is None:
            embeddings = await self._llm.embed([content])
            if not embeddings or not embeddings[0]:
                raise ValueError("No embedding data received from LLM client.")
            embedding = embeddings[0]

        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        metadata_payload = self._sanitize_jsonish(dict(metadata or {}))
        session_id = get_current_session_id()
        if session_id and "session_id" not in metadata_payload:
            metadata_payload["session_id"] = session_id

        query = text("""
            INSERT INTO embeddings (artifact_id, artifact_type, text, embedding, metadata)
            VALUES (:artifact_id, :artifact_type, :text, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
            RETURNING CAST(id AS text)
        """)

        async with self._conn.session() as session:
            result = await session.execute(query, {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "text": content,
                "embedding": embedding_str,
                "metadata": json.dumps(metadata_payload, ensure_ascii=False),
            })
            row = result.fetchone()
            return str(row[0]) if row else ""

    async def search(
        self,
        query_text: str,
        top_k: int = 10,
        artifact_type: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Semantic search over stored embeddings.

        Embeds the query text and finds the closest matches
        using cosine distance.

        Args:
            query_text: The search query.
            top_k: Number of results to return.
            artifact_type: Optional filter by artifact type.

        Returns:
            List of dicts with 'artifact_id', 'artifact_type', 'text',
            'score', and 'metadata' keys, ordered by relevance.
        """
        query_embeddings = await self._llm.embed([query_text])
        query_embedding = query_embeddings[0]
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        type_filter = ""
        params: dict[str, Any] = {
            "embedding": embedding_str,
            "top_k": top_k,
        }
        if artifact_type:
            type_filter = "AND artifact_type = :artifact_type"
            params["artifact_type"] = artifact_type
        metadata_clauses: list[str] = []
        filters = dict(metadata_filters or {})
        session_id = get_current_session_id()
        if session_id and "session_id" not in filters:
            filters["session_id"] = session_id
        for index, (key, value) in enumerate(filters.items()):
            param_name = f"metadata_{index}"
            metadata_clauses.append(f"AND metadata ->> '{key}' = :{param_name}")
            params[param_name] = str(value)

        query = text(f"""
            SELECT artifact_id, artifact_type, text, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM embeddings
            WHERE 1=1 {type_filter} {' '.join(metadata_clauses)}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        async with self._conn.session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()

        return [
            {
                "artifact_id": row[0],
                "artifact_type": row[1],
                "text": row[2],
                "metadata": row[3],
                "score": float(row[4]) if row[4] else 0.0,
            }
            for row in rows
        ]

    async def search_by_type(
        self,
        query_text: str,
        artifact_types: list[str],
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search across multiple artifact types.

        Useful when an agent wants to find relevant notes AND
        hypotheses AND findings for a given topic.
        """
        query_embeddings = await self._llm.embed([query_text])
        query_embedding = query_embeddings[0]
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        params: dict[str, Any] = {
            "embedding": embedding_str,
            "types": artifact_types,
            "top_k": top_k,
        }
        metadata_clauses: list[str] = []
        filters = dict(metadata_filters or {})
        session_id = get_current_session_id()
        if session_id and "session_id" not in filters:
            filters["session_id"] = session_id
        for index, (key, value) in enumerate(filters.items()):
            param_name = f"metadata_{index}"
            metadata_clauses.append(f"AND metadata ->> '{key}' = :{param_name}")
            params[param_name] = str(value)

        query = text(f"""
            SELECT artifact_id, artifact_type, text, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM embeddings
            WHERE artifact_type = ANY(:types)
              {' '.join(metadata_clauses)}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        async with self._conn.session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()

        return [
            {
                "artifact_id": row[0],
                "artifact_type": row[1],
                "text": row[2],
                "metadata": row[3],
                "score": float(row[4]) if row[4] else 0.0,
            }
            for row in rows
        ]
