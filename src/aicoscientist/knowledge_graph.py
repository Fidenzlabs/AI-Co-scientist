"""networkx-backed knowledge graph with citation provenance.

The graph stores concepts as nodes and relations as edges. Every node and edge keeps
the set of ``Citation`` ids that support it, so provenance is preserved through merges
and deduplication. Merging is additive: concepts with the same slug are unified and
their source ids / domains are accumulated rather than overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .models import (
    Concept,
    DomainSubgraph,
    KnowledgeGraphMetadata,
    Relation,
    slugify,
)


class KnowledgeGraph:
    """A unified, provenance-preserving knowledge graph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    # ──────────────────────── mutation ────────────────────────

    def add_concept(self, concept: Concept) -> None:
        cid = concept.id or slugify(concept.name)
        if self.graph.has_node(cid):
            node = self.graph.nodes[cid]
            node["source_ids"] = sorted(set(node["source_ids"]) | set(concept.source_ids))
            node["domains"] = sorted(set(node["domains"]) | set(concept.domains))
            if not node.get("description") and concept.description:
                node["description"] = concept.description
            if node.get("type", "concept") == "concept" and concept.type != "concept":
                node["type"] = concept.type
        else:
            self.graph.add_node(
                cid,
                name=concept.name,
                type=concept.type,
                description=concept.description,
                domains=list(concept.domains),
                source_ids=list(concept.source_ids),
            )

    def add_relation(self, relation: Relation) -> None:
        s, t = relation.source_id, relation.target_id
        if not self.graph.has_node(s) or not self.graph.has_node(t):
            # Skip dangling edges; concepts must exist first.
            return
        # Deduplicate edges by (source, target, relation type): merge provenance.
        key = relation.relation
        if self.graph.has_edge(s, t, key=key):
            data = self.graph.edges[s, t, key]
            data["source_ids"] = sorted(set(data["source_ids"]) | set(relation.source_ids))
            if not data.get("description") and relation.description:
                data["description"] = relation.description
        else:
            self.graph.add_edge(
                s,
                t,
                key=key,
                relation=relation.relation,
                description=relation.description,
                source_ids=list(relation.source_ids),
            )

    def merge_subgraph(self, sub: DomainSubgraph) -> None:
        """Merge one domain swarm's subgraph into the unified graph."""
        for concept in sub.concepts:
            if sub.domain not in concept.domains:
                concept.domains.append(sub.domain)
            self.add_concept(concept)
        for relation in sub.relations:
            self.add_relation(relation)

    # ──────────────────────── queries ────────────────────────

    def concepts(self) -> list[Concept]:
        return [
            Concept(
                id=cid,
                name=data["name"],
                type=data.get("type", "concept"),
                description=data.get("description", ""),
                domains=data.get("domains", []),
                source_ids=data.get("source_ids", []),
            )
            for cid, data in self.graph.nodes(data=True)
        ]

    def relations(self) -> list[Relation]:
        out: list[Relation] = []
        for s, t, data in self.graph.edges(data=True):
            out.append(
                Relation(
                    source_id=s,
                    target_id=t,
                    relation=data.get("relation", "related_to"),
                    description=data.get("description", ""),
                    source_ids=data.get("source_ids", []),
                )
            )
        return out

    def metadata(self) -> KnowledgeGraphMetadata:
        domains: set[str] = set()
        types: dict[str, int] = {}
        all_sources: set[str] = set()
        for _, data in self.graph.nodes(data=True):
            domains.update(data.get("domains", []))
            t = data.get("type", "concept")
            types[t] = types.get(t, 0) + 1
            all_sources.update(data.get("source_ids", []))
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        density = (m / (n * (n - 1))) if n > 1 else 0.0
        return KnowledgeGraphMetadata(
            num_concepts=n,
            num_relations=m,
            num_citations=len(all_sources),
            domains=sorted(domains),
            density=round(density, 4),
            concept_types=types,
        )

    def degree_centrality(self) -> dict[str, float]:
        """Centrality used as a novelty/importance signal for hypotheses."""
        if self.graph.number_of_nodes() == 0:
            return {}
        simple = nx.DiGraph(self.graph)
        return nx.degree_centrality(simple)

    # ──────────────────────── serialization ────────────────────────

    def to_dict(self) -> dict:
        return {
            "concepts": [c.model_dump() for c in self.concepts()],
            "relations": [r.model_dump() for r in self.relations()],
        }

    def save(self, json_path: Path, graphml_path: Path | None = None) -> None:
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        if graphml_path is not None:
            self._save_graphml(graphml_path)

    def _save_graphml(self, graphml_path: Path) -> None:
        # GraphML can't hold list attributes; flatten them to strings on a copy.
        flat = nx.MultiDiGraph()
        for cid, data in self.graph.nodes(data=True):
            flat.add_node(
                cid,
                name=data.get("name", cid),
                type=data.get("type", "concept"),
                description=data.get("description", ""),
                domains=",".join(data.get("domains", [])),
                source_ids=",".join(data.get("source_ids", [])),
            )
        for s, t, data in self.graph.edges(data=True):
            flat.add_edge(
                s,
                t,
                relation=data.get("relation", "related_to"),
                description=data.get("description", ""),
                source_ids=",".join(data.get("source_ids", [])),
            )
        nx.write_graphml(flat, graphml_path)

    @classmethod
    def from_subgraphs(cls, subgraphs: list[DomainSubgraph]) -> "KnowledgeGraph":
        kg = cls()
        for sub in subgraphs:
            kg.merge_subgraph(sub)
        return kg
