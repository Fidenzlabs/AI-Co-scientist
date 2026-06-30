"""Crossref source client (150M+ works, DOIs and citation counts, no key required)."""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation
from .base import clean_abstract, make_client

_BASE = "https://api.crossref.org/works"


class CrossrefClient:
    name = "crossref"

    def search(self, query: str, limit: int) -> list[Citation]:
        settings = get_settings()
        params = {
            "query": query,
            "rows": limit,
            "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count,URL",
            "mailto": settings.contact_email,
        }
        with make_client() as client:
            resp = client.get(_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("message", {}).get("items", [])
        return [self._to_citation(it) for it in items]

    def _to_citation(self, item: dict) -> Citation:
        doi = item.get("DOI", "")
        title_list = item.get("title") or [""]
        title = title_list[0] if title_list else ""
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in item.get("author", [])
        ]
        year = None
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        if issued and issued[0] and issued[0][0]:
            year = issued[0][0]
        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else None
        abstract = item.get("abstract")
        if abstract:
            # Crossref abstracts are JATS XML; strip tags crudely.
            import re

            abstract = re.sub(r"<[^>]+>", " ", abstract)
        return Citation(
            id=Citation.make_id(self.name, doi or title[:40]),
            source=self.name,
            title=title,
            authors=[a for a in authors if a.strip()],
            year=year,
            venue=venue,
            doi=doi or None,
            url=item.get("URL"),
            abstract=clean_abstract(abstract),
            citation_count=item.get("is-referenced-by-count"),
        )
