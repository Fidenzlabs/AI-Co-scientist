"""Semantic Scholar source client (Graph API).

Works without a key but is aggressively rate-limited; an optional
``SEMANTIC_SCHOLAR_API_KEY`` raises those limits when present.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation
from .base import clean_abstract, make_client

_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,authors,venue,externalIds,citationCount,url"


class SemanticScholarClient:
    name = "semantic_scholar"

    def search(self, query: str, limit: int) -> list[Citation]:
        settings = get_settings()
        params = {"query": query, "limit": limit, "fields": _FIELDS}
        with make_client() as client:
            headers = {}
            if settings.semantic_scholar_api_key:
                headers["x-api-key"] = settings.semantic_scholar_api_key
            resp = client.get(_BASE, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [self._to_citation(p) for p in data.get("data", [])]

    def _to_citation(self, paper: dict) -> Citation:
        native = paper.get("paperId", "")
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext = paper.get("externalIds") or {}
        return Citation(
            id=Citation.make_id(self.name, native),
            source=self.name,
            title=paper.get("title", "") or "",
            authors=[a for a in authors if a],
            year=paper.get("year"),
            venue=paper.get("venue") or None,
            doi=ext.get("DOI"),
            url=paper.get("url"),
            abstract=clean_abstract(paper.get("abstract")),
            citation_count=paper.get("citationCount"),
        )
