"""Typed artifacts shared across Layers 1 and 2.

These pydantic models are the contract between the source clients, the agents, the
knowledge graph, and the persistence layer. Keeping them strongly typed lets the LLM
return structured output that flows through the pipeline without ad-hoc dicts.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    """Stable, normalized identifier for a concept name (used for dedup)."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


# ──────────────────────────── Citations ────────────────────────────


class Citation(BaseModel):
    """A normalized reference to a scholarly source."""

    id: str = Field(description="Stable provenance id, e.g. 'arxiv:2401.00001'")
    source: str = Field(description="Originating API: arxiv|openalex|crossref|pubmed|semantic_scholar|mock")
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None

    @staticmethod
    def make_id(source: str, native_id: str) -> str:
        native = native_id.strip() or hashlib.sha1(f"{source}".encode()).hexdigest()[:10]
        return f"{source}:{native}"

    def short(self) -> str:
        author = self.authors[0] + " et al." if self.authors else "Unknown"
        year = f" ({self.year})" if self.year else ""
        return f"{author}{year}. {self.title}".strip()


# ──────────────────────────── Knowledge graph ────────────────────────────


class Concept(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(description="slug of the concept name")
    name: str
    type: str = Field(default="concept", description="entity type, e.g. gene, drug, method, disease, concept")
    description: str = ""
    domains: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list, description="Citation ids supporting this concept")

    @classmethod
    def new(cls, name: str, **kwargs) -> "Concept":
        return cls(id=slugify(name), name=name.strip(), **kwargs)


class Relation(BaseModel):
    """A directed, typed edge between two concepts."""

    source_id: str = Field(description="slug of source concept")
    target_id: str = Field(description="slug of target concept")
    relation: str = Field(default="related_to", description="relationship type, e.g. inhibits, causes, treats, associated_with")
    description: str = ""
    source_ids: list[str] = Field(default_factory=list, description="Citation ids supporting this relation")


class DomainSubgraph(BaseModel):
    """One research swarm's evolving state graph for a single domain."""

    domain: str
    keywords: list[str] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""


# ──────────────────────────── Hypotheses ────────────────────────────


class Evidence(BaseModel):
    """A single piece of supporting or contradicting evidence."""

    statement: str
    stance: Literal["supporting", "contradicting"] = "supporting"
    source_ids: list[str] = Field(default_factory=list)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisStateGraph(BaseModel):
    """The dedicated state graph attached to each hypothesis."""

    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    related_concept_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    references: list[str] = Field(default_factory=list, description="Citation ids")


class RankingScores(BaseModel):
    """The components of a hypothesis ranking."""

    evidence_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    """A competing scientific hypothesis with its state graph and reasoning trace."""

    id: str
    statement: str
    rationale: str = ""
    novelty_assessment: str = ""
    reasoning_trace: list[str] = Field(default_factory=list)
    state_graph: HypothesisStateGraph = Field(default_factory=HypothesisStateGraph)
    scores: RankingScores = Field(default_factory=RankingScores)

    @staticmethod
    def make_id(index: int) -> str:
        return f"H{index + 1:02d}"


# ──────────────────────────── Aggregate outputs ────────────────────────────


class KnowledgeGraphMetadata(BaseModel):
    num_concepts: int = 0
    num_relations: int = 0
    num_citations: int = 0
    domains: list[str] = Field(default_factory=list)
    density: float = 0.0
    concept_types: dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=_now)


class ResearchProvenance(BaseModel):
    """Audit trail of how Layer 1 reached its conclusions."""

    idea: str
    run_id: str
    domains: list[str] = Field(default_factory=list)
    sources_queried: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_now)


class Layer1Output(BaseModel):
    """Complete persistent output of the Deep Research Engine."""

    run_id: str
    idea: str
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    kg_metadata: KnowledgeGraphMetadata = Field(default_factory=KnowledgeGraphMetadata)
    provenance: ResearchProvenance | None = None

    def top(self, n: int = 5) -> list[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.scores.composite, reverse=True)[:n]


# ──────────────────────────── Layer 2 ────────────────────────────


class ResearcherDecision(BaseModel):
    """Structured capture of the human-in-the-loop choice."""

    action: Literal["select", "modify", "merge", "new", "quit"]
    selected_ids: list[str] = Field(default_factory=list)
    statement: str | None = Field(
        default=None, description="Edited/merged/new hypothesis statement"
    )
    notes: str = ""
    decided_at: str = Field(default_factory=_now)


class OfficialHypothesis(BaseModel):
    """The official research hypothesis produced by Layer 2."""

    run_id: str
    statement: str
    origin: ResearcherDecision
    state_graph: HypothesisStateGraph = Field(default_factory=HypothesisStateGraph)
    source_hypothesis_ids: list[str] = Field(default_factory=list)
    finalized_at: str = Field(default_factory=_now)
