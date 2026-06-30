"""arXiv source client (Atom XML API, no key required)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..models import Citation
from .base import clean_abstract, make_client

_ATOM = "http://www.w3.org/2005/Atom"
_NS = {"atom": _ATOM}
_BASE = "http://export.arxiv.org/api/query"


class ArxivClient:
    name = "arxiv"

    def search(self, query: str, limit: int) -> list[Citation]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        with make_client() as client:
            resp = client.get(_BASE, params=params)
            resp.raise_for_status()
        return self._parse(resp.text)

    def _parse(self, xml_text: str) -> list[Citation]:
        root = ET.fromstring(xml_text)
        citations: list[Citation] = []
        for entry in root.findall("atom:entry", _NS):
            raw_id = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
            native = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
            title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
            summary = entry.findtext("atom:summary", default="", namespaces=_NS)
            published = entry.findtext("atom:published", default="", namespaces=_NS) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
                for a in entry.findall("atom:author", _NS)
            ]
            citations.append(
                Citation(
                    id=Citation.make_id(self.name, native),
                    source=self.name,
                    title=title,
                    authors=[a for a in authors if a],
                    year=year,
                    url=raw_id or None,
                    abstract=clean_abstract(summary),
                )
            )
        return citations
