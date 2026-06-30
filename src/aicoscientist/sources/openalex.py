"""OpenAlex source client (250M+ works, no key required)."""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation
from .base import clean_abstract, make_client

_BASE = "https://api.openalex.org/works"


class OpenAlexClient:
    name = "openalex"

    def search(self, query: str, limit: int) -> list[Citation]:
        settings = get_settings()
        params = {
            "search": query,
            "per_page": limit,
            "sort": "relevance_score:desc",
            "mailto": settings.contact_email,
        }
        with make_client() as client:
            resp = client.get(_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [self._to_citation(w) for w in data.get("results", [])]

    def _to_citation(self, work: dict) -> Citation:
        native = (work.get("id") or "").rsplit("/", 1)[-1]
        authors = [
            (a.get("author", {}) or {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        primary = work.get("primary_location") or {}
        venue = (primary.get("source") or {}).get("display_name")
        return Citation(
            id=Citation.make_id(self.name, native),
            source=self.name,
            title=work.get("title") or work.get("display_name") or "",
            authors=[a for a in authors if a],
            year=work.get("publication_year"),
            venue=venue,
            doi=(work.get("doi") or "").replace("https://doi.org/", "") or None,
            url=work.get("id"),
            abstract=clean_abstract(self._reconstruct_abstract(work)),
            citation_count=work.get("cited_by_count"),
        )

    @staticmethod
    def _reconstruct_abstract(work: dict) -> str | None:
        """OpenAlex stores abstracts as an inverted index; rebuild the text."""
        inv = work.get("abstract_inverted_index")
        if not inv:
            return None
        positions: list[tuple[int, str]] = []
        for word, idxs in inv.items():
            for i in idxs:
                positions.append((i, word))
        positions.sort()
        return " ".join(word for _, word in positions)
