"""Lightweight schemas used for LLM structured output inside agents.

These are deliberately simpler than the persistent ``models`` (names instead of slugs,
flat fields) so the LLM has an easy target. Agents translate them into the richer
domain models afterwards.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DomainPlan(BaseModel):
    domain: str = Field(description="Short name of the research domain / keyword cluster")
    keywords: list[str] = Field(description="3-6 search keywords/phrases for this domain")


class DomainDecomposition(BaseModel):
    domains: list[DomainPlan] = Field(description="Distinct research domains for the idea")


class ConceptOut(BaseModel):
    name: str
    type: str = Field(default="concept", description="e.g. gene, drug, disease, method, concept")
    description: str = ""


class RelationOut(BaseModel):
    source: str = Field(description="source concept name")
    target: str = Field(description="target concept name")
    relation: str = Field(default="related_to", description="e.g. inhibits, treats, causes, associated_with")
    description: str = ""


class ExtractionResult(BaseModel):
    concepts: list[ConceptOut] = Field(default_factory=list)
    relations: list[RelationOut] = Field(default_factory=list)


class EvidenceOut(BaseModel):
    statement: str
    source_ids: list[str] = Field(default_factory=list)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisDraft(BaseModel):
    statement: str = Field(description="A specific, testable scientific hypothesis")
    rationale: str = ""
    novelty_assessment: str = ""
    reasoning_trace: list[str] = Field(default_factory=list)
    supporting_evidence: list[EvidenceOut] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceOut] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list, description="concept names")
    references: list[str] = Field(default_factory=list, description="citation ids")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisDrafts(BaseModel):
    hypotheses: list[HypothesisDraft] = Field(default_factory=list)
