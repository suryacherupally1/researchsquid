"""
Hierarchical document summarization.

Generates multi-level summaries of documents:
  Document → Section → Subsection

This gives agents a "table of contents" view of large documents
so they can understand structure before deep-diving into specific
chunks via RAG.
"""

import asyncio
from src.llm.client import LLMClient
from src.models.note import Note


class HierarchicalSummarizer:
    """
    Produces layered summaries of a document's sections.

    First summarizes each section individually, then produces an
    overall document summary from the section summaries. All
    summaries are stored as Notes in the knowledge graph.

    Usage:
        summarizer = HierarchicalSummarizer(llm_client)
        notes = await summarizer.summarize(sections, source_id, agent_id)
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def summarize(
        self,
        sections: list[dict[str, str]],
        source_id: str,
        agent_id: str,
    ) -> list[Note]:
        """
        Generate hierarchical summaries for a document.

        Args:
            sections: List of dicts with 'text' and 'section_title'.
            source_id: ID of the parent Source.
            agent_id: ID of the agent performing summarization.

        Returns:
            List of Note models — one per section plus one overall summary.
        """
        notes: list[Note] = []
        
        # Limit concurrent LLM calls to prevent rate limiting
        semaphore = asyncio.Semaphore(10)
        
        async def _process_section(section: dict[str, str]) -> tuple[str, Note | None]:
            text = section.get("text", "")
            title = section.get("section_title", "")

            if not text.strip() or len(text) < 100:
                # Too short to summarize — use as-is
                return text, None

            async with semaphore:
                summary = await self._summarize_section(text, title)

            note = Note(
                text=f"[Section Summary: {title}] {summary}" if title else f"[Section Summary] {summary}",
                source_chunk_ids=[],  # Will be linked to chunks separately
                created_by=agent_id,
                confidence=0.8,
                provenance=[source_id],
                tags=["summary", "section"],
            )
            return summary, note

        tasks = [_process_section(sec) for sec in sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        section_summaries: list[str] = []
        for res in results:
            if isinstance(res, Exception):
                continue
            summary, note = res
            section_summaries.append(summary)
            if note:
                notes.append(note)

        # Generate overall document summary from section summaries
        if section_summaries:
            overall = await self._summarize_document(section_summaries)
            doc_note = Note(
                text=f"[Document Summary] {overall}",
                source_chunk_ids=[],
                created_by=agent_id,
                confidence=0.8,
                provenance=[source_id],
                tags=["summary", "document"],
            )
            notes.append(doc_note)

        return notes

    async def _summarize_section(self, text: str, title: str) -> str:
        """Summarize a single section of a document."""
        # Truncate very long sections to fit context
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        prompt = (
            f"Summarize the following section concisely, preserving key "
            f"claims, data, and methodology.\n\n"
            f"Section: {title}\n\n{text}"
        )
        return await self._llm.complete(
            prompt,
            system="You are a research assistant. Produce concise, accurate summaries.",
            temperature=0.3,
            max_tokens=500,
        )

    async def _summarize_document(self, section_summaries: list[str]) -> str:
        """Produce an overall document summary from section summaries."""
        combined = "\n\n".join(
            f"- {s}" for s in section_summaries if s.strip()
        )
        prompt = (
            f"Given these section summaries, write a comprehensive "
            f"2-3 paragraph document summary:\n\n{combined}"
        )
        return await self._llm.complete(
            prompt,
            system="You are a research assistant. Produce clear, comprehensive summaries.",
            temperature=0.3,
            max_tokens=800,
        )
