"""Parent orchestration agent.

Responsibilities (from the architecture spec):
- Creates and manages research swarms by decomposing the idea into domains.
- Merges individual domain state graphs into a unified knowledge graph (dedup +
  provenance handled by ``KnowledgeGraph``).
- Ranks hypotheses using evidence quality, novelty, consistency, and confidence.
"""

from __future__ import annotations

import logging
import re

from ..config import get_settings
from ..knowledge_graph import KnowledgeGraph
from ..llm import structured_call
from ..models import DomainSubgraph, Hypothesis, RankingScores, slugify
from .schemas import DomainDecomposition

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "You are the orchestration agent of an AI co-scientist specialized in "
    "area-selective atomic layer deposition (AS-ALD) surface chemistry. Decompose a "
    "research idea into distinct, complementary research domains (keyword clusters). "
    "Cover the AS-ALD problem structure: (a) amorphous surface models and site "
    "densities, (b) small-molecule inhibitors / passivation chemistry, (c) ALD "
    "precursors and film growth, (d) reaction mechanisms (chemisorption vs "
    "physisorption, barriers) and measured selectivity. Provide focused search "
    "keywords per domain using surface-chemistry vocabulary."
)


class Orchestrator:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    # ──────────────────────── decompose ────────────────────────

    def decompose(self, idea: str) -> list[dict]:
        settings = get_settings()
        max_domains = settings.max_domains
        if self.offline:
            return self._decompose_heuristic(idea, max_domains)

        user = (
            f"Research idea: {idea}\n\n"
            f"Produce up to {max_domains} research domains with keywords."
        )
        try:
            result = structured_call(DomainDecomposition, _DECOMPOSE_SYSTEM, user)
            domains = [
                {"domain": d.domain.strip(), "keywords": [k for k in d.keywords if k.strip()]}
                for d in result.domains
                if d.domain.strip()
            ][:max_domains]
            if domains:
                return domains
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM decomposition failed: %s", exc)
        return self._decompose_heuristic(idea, max_domains)

    def _decompose_heuristic(self, idea: str, max_domains: int) -> list[dict]:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", idea)]
        seen, keywords = set(), []
        for w in words:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                keywords.append(w)
        keywords = keywords or [idea]
        angles = [
            ("Amorphous surface models", keywords[:2] + [
                "amorphous silica surface model", "silanol density",
                "amorphous silicon nitride surface", "site density"]),
            ("Inhibitors and passivation", keywords[:2] + [
                "small molecule inhibitor ALD", "area-selective deposition passivation",
                "chemisorption inhibitor", "self-assembled monolayer blocking"]),
            ("Precursors and film growth", keywords[:2] + [
                "ALD precursor half-reaction", "BDEAS silicon oxide ALD",
                "nucleation delay growth per cycle", "selective film growth"]),
            ("Selectivity mechanisms", keywords[:2] + [
                "adsorption energy DFT surface", "reaction barrier selectivity",
                "chemisorption physisorption selectivity", "area-selective ALD selectivity"]),
        ]
        domains = []
        for name, kws in angles[:max_domains]:
            domains.append({"domain": name, "keywords": [idea] + kws[:5]})
        return domains

    # ──────────────────────── merge ────────────────────────

    def merge(self, subgraphs: list[DomainSubgraph]) -> KnowledgeGraph:
        kg = KnowledgeGraph.from_subgraphs(subgraphs)
        logger.info(
            "merged %d subgraphs -> %d concepts, %d relations",
            len(subgraphs),
            kg.graph.number_of_nodes(),
            kg.graph.number_of_edges(),
        )
        return kg

    # ──────────────────────── rank ────────────────────────

    def rank(
        self, hypotheses: list[Hypothesis], kg: KnowledgeGraph
    ) -> list[Hypothesis]:
        """Score each hypothesis on evidence quality, novelty, consistency, confidence."""
        centrality = kg.degree_centrality()
        all_concepts = {c.id for c in kg.concepts()}

        for h in hypotheses:
            sg = h.state_graph
            support = sg.supporting_evidence
            contra = sg.contradicting_evidence

            # Evidence quality: volume + strength of supporting evidence, with sources.
            sup_strength = sum(e.strength for e in support)
            evidence_quality = _squash(sup_strength + 0.3 * len({s for e in support for s in e.source_ids}))

            # Consistency: supporting outweighs contradicting evidence.
            total = len(support) + len(contra)
            consistency = (len(support) / total) if total else 0.4

            # Novelty: hypotheses touching less-central concepts are more novel; also
            # reward hypotheses grounded in known concepts at all.
            related = [c for c in sg.related_concept_ids if c in all_concepts]
            if related:
                avg_central = sum(centrality.get(c, 0.0) for c in related) / len(related)
                novelty = _squash(1.0 - avg_central) * (0.6 + 0.4 * min(1.0, len(related) / 3))
            else:
                novelty = 0.4

            confidence = sg.confidence

            composite = (
                0.30 * evidence_quality
                + 0.25 * novelty
                + 0.20 * consistency
                + 0.25 * confidence
            )
            h.scores = RankingScores(
                evidence_quality=round(evidence_quality, 3),
                novelty=round(novelty, 3),
                consistency=round(consistency, 3),
                confidence=round(confidence, 3),
                composite=round(composite, 3),
            )

        return sorted(hypotheses, key=lambda h: h.scores.composite, reverse=True)


def _squash(x: float) -> float:
    """Clamp a non-negative score into [0, 1] with diminishing returns."""
    if x <= 0:
        return 0.0
    return min(1.0, x / (1.0 + x) * 2)
