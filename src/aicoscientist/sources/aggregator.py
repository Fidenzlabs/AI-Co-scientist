"""Aggregates the scholarly source clients with graceful per-source fallback.

A single source failing (network error, rate limit, malformed response) must never
abort a run, so each client is wrapped in try/except and simply contributes nothing on
failure. Results across sources are deduplicated by DOI and by normalized title.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import get_settings
from ..models import Citation, slugify
from .arxiv import ArxivClient
from .crossref import CrossrefClient
from .mock import MockClient
from .openalex import OpenAlexClient
from .pubmed import PubMedClient
from .semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class SourceAggregator:
    """Queries every enabled source and returns a deduplicated citation list."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline
        if offline:
            self.clients = [MockClient()]
        else:
            self.clients = [
                ArxivClient(),
                OpenAlexClient(),
                CrossrefClient(),
                PubMedClient(),
                SemanticScholarClient(),
            ]

    @property
    def source_names(self) -> list[str]:
        return [c.name for c in self.clients]

    def search(self, query: str, limit: int | None = None) -> list[Citation]:
        settings = get_settings()
        limit = limit or settings.max_results_per_source
        collected: list[Citation] = []

        with ThreadPoolExecutor(max_workers=max(1, len(self.clients))) as pool:
            futures = {
                pool.submit(self._safe_search, client, query, limit): client.name
                for client in self.clients
            }
            for future in as_completed(futures):
                collected.extend(future.result())

        return self._dedupe(collected)

    def _safe_search(self, client, query: str, limit: int) -> list[Citation]:
        try:
            results = client.search(query, limit)
            logger.info("source %s returned %d results", client.name, len(results))
            return results
        except Exception as exc:  # noqa: BLE001 - degrade gracefully per source
            logger.warning("source %s failed: %s", client.name, exc)
            return []

    @staticmethod
    def _dedupe(citations: list[Citation]) -> list[Citation]:
        by_doi: dict[str, Citation] = {}
        by_title: dict[str, Citation] = {}
        out: list[Citation] = []
        for c in citations:
            if c.doi:
                key = c.doi.lower()
                if key in by_doi:
                    continue
                by_doi[key] = c
            title_key = slugify(c.title)[:80]
            if title_key and title_key in by_title:
                continue
            if title_key:
                by_title[title_key] = c
            out.append(c)
        return out
