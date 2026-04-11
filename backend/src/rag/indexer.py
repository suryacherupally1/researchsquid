"""
RAG indexer — the pipeline from raw sources to embedded, retrievable artifacts.

Coordinates ingestion, chunking, summarization, graph storage, and
embedding in a single pipeline. This is the main entry point for
adding new research material to the institute.
"""

import asyncio
from typing import Any
from src.graph.repository import GraphRepository
from src.ingest.chunker import TextChunker
from src.ingest.pdf import PDFIngestor
from src.ingest.url import URLIngestor
from src.ingest.text import TextIngestor
from src.ingest.summarizer import HierarchicalSummarizer
from src.rag.store import VectorStore
from src.events.bus import EventBus
from src.models.events import Event, EventType
from src.models.source import Source


class RAGIndexer:
    """
    End-to-end pipeline for ingesting sources into the knowledge graph.

    Handles the full flow:
    1. Read source (PDF/URL/text)
    2. Chunk into segments
    3. Generate hierarchical summaries
    4. Store everything in Neo4j
    5. Embed text in pgvector

    Usage:
        indexer = RAGIndexer(graph_repo, vector_store, llm, event_bus)
        source_id = await indexer.ingest_pdf("paper.pdf", "agent-1")
        source_id = await indexer.ingest_url("https://example.com", "agent-1")
    """

    def __init__(
        self,
        graph: GraphRepository,
        vector_store: VectorStore,
        summarizer: HierarchicalSummarizer,
        event_bus: EventBus,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._summarizer = summarizer
        self._bus = event_bus
        self._chunker = TextChunker()
        self._pdf_ingestor = PDFIngestor()
        self._url_ingestor = URLIngestor()
        self._text_ingestor = TextIngestor()

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return str(value or "").replace("\x00", "")

    @classmethod
    def _sanitize_sections(
        cls,
        sections: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        sanitized: list[dict[str, str]] = []
        for section in sections:
            sanitized.append({
                str(key): cls._sanitize_text(value)
                for key, value in section.items()
            })
        return sanitized

    async def ingest_pdf(self, file_path: str, agent_id: str) -> str:
        """Ingest a PDF file into the knowledge graph."""
        source, pages = await self._pdf_ingestor.ingest(file_path, agent_id)
        return await self._process_source(source, pages, agent_id)

    async def ingest_url(self, url: str, agent_id: str) -> str:
        """Ingest a web page into the knowledge graph."""
        source, sections = await self._url_ingestor.ingest(url, agent_id)
        return await self._process_source(source, sections, agent_id)

    async def ingest_text(
        self,
        text: str,
        agent_id: str,
        title: str = "Untitled",
    ) -> str:
        """Ingest raw text into the knowledge graph."""
        source, sections = await self._text_ingestor.ingest_text(
            text, agent_id, title=title
        )
        return await self._process_source(source, sections, agent_id)

    async def ingest_file(self, file_path: str, agent_id: str) -> str:
        """Ingest a text/markdown file into the knowledge graph."""
        source, sections = await self._text_ingestor.ingest_file(
            file_path, agent_id
        )
        return await self._process_source(source, sections, agent_id)

    async def index_artifact(self, artifact: Any) -> str | None:
        """Embed an unsupported standalone artifact like a Finding."""
        if not hasattr(artifact, "text") or not artifact.text:
            return None
        
        artifact_type = artifact.__class__.__name__.lower()
        embedding_id = await self._vector_store.store_embedding(
            artifact_id=artifact.id,
            artifact_type=artifact_type,
            content=artifact.text,
            metadata={"tags": getattr(artifact, "tags", [])},
        )
        if hasattr(self._graph, "update"):
            await self._graph.update(artifact.id, {"embedding_id": embedding_id})
        return embedding_id

    async def _process_source(
        self,
        source: Source,
        sections: list[dict[str, str]],
        agent_id: str,
    ) -> str:
        """
        Common processing pipeline for all source types.

        1. Store Source node in Neo4j
        2. Chunk the text
        3. Store SourceChunk nodes + link to Source
        4. Embed all chunks in pgvector
        5. Generate hierarchical summaries
        6. Store summary Notes + embed them

        Returns:
            The Source's ID.
        """
        source.title = self._sanitize_text(source.title)
        source.summary = self._sanitize_text(source.summary)
        source.file_path = self._sanitize_text(source.file_path)
        source.uri = self._sanitize_text(source.uri)
        sections = self._sanitize_sections(sections)

        # 1. Store source in graph
        await self._graph.create(source)

        # 2. Chunk the text
        chunks = self._chunker.chunk(sections, source.id, agent_id)

        # 3. Store chunks and link to source in DB (fast/sync relative to API)
        for chunk in chunks:
            chunk.text = self._sanitize_text(chunk.text)
            chunk.section_title = self._sanitize_text(chunk.section_title)
            await self._graph.create(chunk)
            await self._graph.link_chunk_to_source(chunk.id, source.id)

        # 4. Embed chunk text concurrently
        async def _embed_chunk(c):
            emb_id = await self._vector_store.store_embedding(
                artifact_id=c.id,
                artifact_type="source_chunk",
                content=c.text,
                metadata={"source_id": source.id, "section": c.section_title},
            )
            await self._graph.update(c.id, {"embedding_id": emb_id})

        embed_tasks = [_embed_chunk(c) for c in chunks]
        await asyncio.gather(*embed_tasks)

        # 5. Generate hierarchical summaries
        summary_notes = await self._summarizer.summarize(
            sections, source.id, agent_id
        )

        # 6. Store and embed summary notes
        for note in summary_notes:
            note.text = self._sanitize_text(note.text)
            await self._graph.create(note)
            await self._graph.create_edge(source.id, note.id, "HAS_SUMMARY")

        # Embed summary notes concurrently
        async def _embed_note(n):
            emb_id = await self._vector_store.store_embedding(
                artifact_id=n.id,
                artifact_type="note",
                content=n.text,
                metadata={"source_id": source.id, "tags": n.tags},
            )
            await self._graph.update(n.id, {"embedding_id": emb_id})

        note_embed_tasks = [_embed_note(n) for n in summary_notes]
        await asyncio.gather(*note_embed_tasks)

        # Emit event
        await self._bus.publish(Event(
            event_type=EventType.SOURCE_INGESTED,
            agent_id=agent_id,
            artifact_id=source.id,
            artifact_type="source",
            payload={
                "title": source.title,
                "source_type": source.source_type,
                "chunks_count": len(chunks),
                "summaries_count": len(summary_notes),
            },
        ))

        return source.id
