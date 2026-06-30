"""PubMed source client via NIH E-utilities (esearch + esummary, no key required)."""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation
from .base import make_client

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubMedClient:
    name = "pubmed"

    def search(self, query: str, limit: int) -> list[Citation]:
        settings = get_settings()
        common = {"db": "pubmed", "retmode": "json", "email": settings.contact_email}
        with make_client() as client:
            search_resp = client.get(
                _ESEARCH, params={**common, "term": query, "retmax": limit}
            )
            search_resp.raise_for_status()
            ids = (
                search_resp.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )
            if not ids:
                return []
            summary_resp = client.get(
                _ESUMMARY, params={**common, "id": ",".join(ids)}
            )
            summary_resp.raise_for_status()
            result = summary_resp.json().get("result", {})

        citations: list[Citation] = []
        for uid in result.get("uids", []):
            paper = result.get(uid, {})
            citations.append(self._to_citation(uid, paper))
        return citations

    def _to_citation(self, uid: str, paper: dict) -> Citation:
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        pubdate = paper.get("pubdate", "") or ""
        year = int(pubdate[:4]) if pubdate[:4].isdigit() else None
        doi = None
        for aid in paper.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value")
        return Citation(
            id=Citation.make_id(self.name, uid),
            source=self.name,
            title=paper.get("title", ""),
            authors=[a for a in authors if a],
            year=year,
            venue=paper.get("fulljournalname") or paper.get("source"),
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            abstract=None,  # esummary omits abstracts; title is enough for extraction
        )
