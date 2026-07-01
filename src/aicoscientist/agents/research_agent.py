"""Domain-specialized research agent (one per swarm subgroup).

Searches the scholarly sources for its domain, then extracts concepts and relations
from the retrieved titles/abstracts into a ``DomainSubgraph``. Every extracted concept
and relation carries the ids of the citations that support it, preserving provenance.
"""

from __future__ import annotations

import logging
import re

from ..llm import structured_call
from ..models import Citation, Concept, DomainSubgraph, Relation, slugify
from ..sources import SourceAggregator
from .schemas import ExtractionResult

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "You are a surface-chemistry knowledge extraction agent for an AS-ALD co-scientist. "
    "Given a research idea and a set of paper titles and abstracts, extract the key "
    "entities as typed concepts: surfaces (material/phase/site_type/site_density), "
    "inhibitors (functional_group/vapor_pressure/removability), precursors "
    "(target_film), mechanisms (chemisorb|physisorb, adsorption energy dE, barrier Ea), "
    "and selectivity_results (film, thickness, % selectivity, method). Extract the "
    "directed relationships between them (e.g. passivates, chemisorbs_on, "
    "physisorbs_on, blocks, grows_on, selective_for, measured_by). Only extract "
    "concepts grounded in the provided text. Be concise and specific."
)

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "study", "studies",
    "using", "based", "between", "into", "their", "these", "those", "have",
    "been", "which", "were", "also", "such", "than", "then", "they", "them",
    "research", "paper", "results", "analysis", "approach", "novel", "effects",
}


class ResearchAgent:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline
        self.aggregator = SourceAggregator(offline=offline)

    def investigate(self, idea: str, domain: str, keywords: list[str]) -> DomainSubgraph:
        queries = keywords or [domain]
        citations: list[Citation] = []
        for q in queries[:4]:
            citations.extend(self.aggregator.search(f"{q}"))
        citations = self._dedupe(citations)

        if not citations:
            return DomainSubgraph(domain=domain, keywords=keywords, notes="no citations found")

        if self.offline:
            concepts, relations = self._extract_heuristic(idea, domain, citations)
        else:
            concepts, relations = self._extract_llm(idea, domain, citations)

        return DomainSubgraph(
            domain=domain,
            keywords=keywords,
            concepts=concepts,
            relations=relations,
            citations=citations,
            notes=f"{len(citations)} citations, {len(concepts)} concepts",
        )

    # ──────────────────────── extraction (LLM) ────────────────────────

    def _extract_llm(
        self, idea: str, domain: str, citations: list[Citation]
    ) -> tuple[list[Concept], list[Relation]]:
        corpus = self._format_corpus(citations)
        user = (
            f"Research idea: {idea}\nDomain: {domain}\n\n"
            f"Papers:\n{corpus}\n\n"
            "Extract concepts and relations as structured data."
        )
        try:
            result = structured_call(ExtractionResult, _EXTRACT_SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM extraction failed for domain %s: %s", domain, exc)
            return self._extract_heuristic(idea, domain, citations)

        all_source_ids = [c.id for c in citations]
        concepts = [
            Concept.new(
                c.name,
                type=c.type or "concept",
                description=c.description,
                domains=[domain],
                source_ids=all_source_ids,
            )
            for c in result.concepts
            if c.name.strip()
        ]
        known = {c.id for c in concepts}
        relations = []
        for r in result.relations:
            s, t = slugify(r.source), slugify(r.target)
            if s in known and t in known and s != t:
                relations.append(
                    Relation(
                        source_id=s,
                        target_id=t,
                        relation=r.relation or "related_to",
                        description=r.description,
                        source_ids=all_source_ids,
                    )
                )
        return concepts, relations

    # ──────────────────────── extraction (offline heuristic) ────────────────────────

    def _extract_heuristic(
        self, idea: str, domain: str, citations: list[Citation]
    ) -> tuple[list[Concept], list[Relation]]:
        """Deterministic keyphrase extraction used in offline mode."""
        term_sources: dict[str, set[str]] = {}
        for c in citations:
            text = f"{c.title}. {c.abstract or ''}"
            for term in self._keyphrases(text):
                term_sources.setdefault(term, set()).add(c.id)

        # Keep the most-supported terms as concepts.
        ranked = sorted(term_sources.items(), key=lambda kv: len(kv[1]), reverse=True)
        top = ranked[:12]
        concepts = [
            Concept.new(
                term,
                description=f"Concept extracted from {domain} literature.",
                domains=[domain],
                source_ids=sorted(srcs),
            )
            for term, srcs in top
        ]
        # Chain consecutive top concepts with co-occurrence relations.
        relations: list[Relation] = []
        for i in range(len(top) - 1):
            (term_a, src_a), (term_b, src_b) = top[i], top[i + 1]
            shared = sorted(src_a & src_b) or sorted(src_a | src_b)
            relations.append(
                Relation(
                    source_id=slugify(term_a),
                    target_id=slugify(term_b),
                    relation="associated_with",
                    description="Co-occurrence in domain literature.",
                    source_ids=shared,
                )
            )
        return concepts, relations

    @staticmethod
    def _keyphrases(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        out = []
        for w in words:
            if w in _STOPWORDS or len(w) < 4:
                continue
            out.append(w)
        # Also capture two-word phrases for a touch more specificity.
        phrases = [f"{a} {b}" for a, b in zip(out, out[1:]) if a != b]
        return out + phrases[:20]

    # ──────────────────────── helpers ────────────────────────

    @staticmethod
    def _format_corpus(citations: list[Citation], limit: int = 20) -> str:
        lines = []
        for c in citations[:limit]:
            abstract = (c.abstract or "")[:600]
            lines.append(f"[{c.id}] {c.title}\n{abstract}".strip())
        return "\n\n".join(lines)

    @staticmethod
    def _dedupe(citations: list[Citation]) -> list[Citation]:
        seen: set[str] = set()
        out: list[Citation] = []
        for c in citations:
            if c.id in seen:
                continue
            seen.add(c.id)
            out.append(c)
        return out
