"""
Arxiv paper search and download.

Agents autonomously search Arxiv for relevant research papers,
download PDFs, and ingest them into the knowledge graph. Uses
the free Arxiv API (no key required).
"""

import os
from pathlib import Path

import httpx

try:
    import arxiv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    arxiv = None

from src.config import Settings, settings as default_settings
from src.events.bus import EventBus
from src.models.events import Event, EventType


class ArxivSearch:
    """
    Search Arxiv for research papers and download PDFs.

    Agents use this to discover academic papers relevant to their
    line of inquiry. Downloaded PDFs are stored in the data/sources
    directory for subsequent ingestion.

    Usage:
        search = ArxivSearch(settings, event_bus)
        papers = await search.search("transformer attention mechanisms")
        pdf_path = await search.download(papers[0]["arxiv_id"])
    """

    def __init__(
        self,
        config: Settings | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config or default_settings
        self._bus = event_bus
        self._download_dir = Path(self._config.data_dir)

    @staticmethod
    def _require_dependency() -> None:
        if arxiv is None:
            raise RuntimeError(
                "The optional 'arxiv' package is not installed. "
                "Install backend extras or add 'arxiv' to the environment "
                "to enable Arxiv search/download."
            )

    async def search(
        self,
        query: str,
        max_results: int = 5,
        agent_id: str = "",
    ) -> list[dict]:
        """
        Search Arxiv for papers matching a query.

        Args:
            query: Natural language or Arxiv-syntax query.
            max_results: Maximum papers to return.
            agent_id: ID of the agent performing the search.

        Returns:
            List of dicts with paper metadata (title, authors,
            abstract, arxiv_id, pdf_url, published date).
        """
        self._require_dependency()
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary,
                "arxiv_id": paper.entry_id.split("/")[-1],
                "pdf_url": paper.pdf_url,
                "published": paper.published.isoformat() if paper.published else "",
                "categories": paper.categories,
            })

        if self._bus and results:
            await self._bus.publish(Event(
                event_type=EventType.SOURCE_DISCOVERED,
                agent_id=agent_id,
                payload={
                    "query": query,
                    "results_count": len(results),
                    "source": "arxiv",
                    "titles": [r["title"] for r in results],
                },
            ))

        return results

    async def download(
        self,
        arxiv_id: str,
        agent_id: str = "",
    ) -> str:
        """
        Download a paper's PDF from Arxiv.

        Args:
            arxiv_id: The Arxiv paper ID (e.g., "2301.07041").
            agent_id: ID of the agent requesting the download.

        Returns:
            Local file path where the PDF was saved.
        """
        self._require_dependency()
        # Ensure download directory exists
        self._download_dir.mkdir(parents=True, exist_ok=True)

        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])

        paper = next(client.results(search))
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        file_path = str(self._download_dir / filename)
        title = paper.title or arxiv_id

        # Download only if not already present
        if not os.path.exists(file_path):
            await self._publish_progress(
                agent_id=agent_id,
                arxiv_id=arxiv_id,
                title=title,
                progress=0,
                stage="starting",
            )
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=120.0,
            ) as client:
                async with client.stream(
                    "GET",
                    paper.pdf_url,
                    headers={"User-Agent": "ResearchSquid/0.1"},
                ) as response:
                    response.raise_for_status()
                    total_bytes = int(response.headers.get("Content-Length") or 0)
                    downloaded = 0
                    last_progress = -1
                    with open(file_path, "wb") as handle:
                        async for chunk in response.aiter_bytes(65536):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            downloaded += len(chunk)
                            progress = (
                                int(downloaded * 100 / total_bytes)
                                if total_bytes > 0
                                else 0
                            )
                            if total_bytes > 0 and progress >= last_progress + 10:
                                last_progress = progress
                                await self._publish_progress(
                                    agent_id=agent_id,
                                    arxiv_id=arxiv_id,
                                    title=title,
                                    progress=min(progress, 95),
                                    stage="downloading",
                                    bytes_downloaded=downloaded,
                                    total_bytes=total_bytes,
                                )
            await self._publish_progress(
                agent_id=agent_id,
                arxiv_id=arxiv_id,
                title=title,
                progress=100,
                stage="downloaded",
            )
        else:
            await self._publish_progress(
                agent_id=agent_id,
                arxiv_id=arxiv_id,
                title=title,
                progress=100,
                stage="cached",
            )

        return file_path

    async def _publish_progress(
        self,
        agent_id: str,
        arxiv_id: str,
        title: str,
        progress: int,
        stage: str,
        bytes_downloaded: int = 0,
        total_bytes: int = 0,
    ) -> None:
        if not self._bus:
            return
        await self._bus.publish(Event(
            event_type=EventType.AGENT_ACTION,
            agent_id=agent_id,
            payload={
                "action": "download_source_progress",
                "source": "arxiv",
                "title": title,
                "arxiv_id": arxiv_id,
                "progress": progress,
                "stage": stage,
                "bytes_downloaded": bytes_downloaded,
                "total_bytes": total_bytes,
            },
        ))
