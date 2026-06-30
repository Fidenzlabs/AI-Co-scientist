"""Layer 1 — Deep Research Engine, as a LangGraph subgraph.

Flow:
    plan_domains  ->  (Send fan-out)  ->  run_domain (xN in parallel)  ->  synthesize

``plan_domains`` decomposes the idea into domain swarms. Each domain is dispatched as a
parallel ``run_domain`` task via the ``Send`` API; results accumulate into ``subgraphs``
through an additive reducer. ``synthesize`` merges the subgraphs into the unified
knowledge graph, generates + ranks competing hypotheses, and persists all artifacts.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.types import Send

from .agents import HypothesisAgent, Orchestrator, ResearchAgent
from .models import (
    DomainSubgraph,
    Layer1Output,
    ResearchProvenance,
)
from .persistence import ArtifactStore
from .sources import SourceAggregator

logger = logging.getLogger(__name__)


class CoScientistState(TypedDict, total=False):
    """Shared state across Layers 1 and 2."""

    idea: str
    offline: bool
    run_id: str

    domains: list[dict]
    subgraphs: Annotated[list[dict], operator.add]
    reasoning_trace: Annotated[list[str], operator.add]
    sources_queried: list[str]

    layer1_output: dict[str, Any]
    top_hypotheses: list[dict]

    # Layer 2
    decision: dict[str, Any]
    official_hypothesis: dict[str, Any]


# ──────────────────────── nodes ────────────────────────


def plan_domains(state: CoScientistState) -> dict:
    offline = state.get("offline", False)
    orchestrator = Orchestrator(offline=offline)
    domains = orchestrator.decompose(state["idea"])
    trace = [f"Decomposed idea into {len(domains)} research domains: " + ", ".join(d["domain"] for d in domains)]
    sources = SourceAggregator(offline=offline).source_names
    return {
        "domains": domains,
        "reasoning_trace": trace,
        "sources_queried": sources,
    }


def fan_out_domains(state: CoScientistState) -> list[Send]:
    """Dispatch one parallel research task per domain swarm."""
    return [
        Send(
            "run_domain",
            {
                "idea": state["idea"],
                "offline": state.get("offline", False),
                "domain": d["domain"],
                "keywords": d.get("keywords", []),
            },
        )
        for d in state["domains"]
    ]


def run_domain(payload: dict) -> dict:
    """Execute a single domain swarm (runs in parallel with its siblings)."""
    agent = ResearchAgent(offline=payload.get("offline", False))
    subgraph = agent.investigate(
        idea=payload["idea"],
        domain=payload["domain"],
        keywords=payload.get("keywords", []),
    )
    trace = [
        f"Swarm '{subgraph.domain}': {len(subgraph.citations)} citations, "
        f"{len(subgraph.concepts)} concepts, {len(subgraph.relations)} relations."
    ]
    return {"subgraphs": [subgraph.model_dump()], "reasoning_trace": trace}


def synthesize(state: CoScientistState) -> dict:
    """Merge subgraphs, generate + rank hypotheses, persist Layer 1 artifacts."""
    offline = state.get("offline", False)
    run_id = state["run_id"]
    orchestrator = Orchestrator(offline=offline)
    hypothesis_agent = HypothesisAgent(offline=offline)

    subgraphs: list[DomainSubgraph] = [
        DomainSubgraph.model_validate(s) for s in state.get("subgraphs", [])
    ]
    kg = orchestrator.merge(subgraphs)

    # Unified citation repository (dedup by id across all swarms).
    citations_by_id = {}
    for sub in subgraphs:
        for c in sub.citations:
            citations_by_id.setdefault(c.id, c)
    citations = list(citations_by_id.values())

    hypotheses = hypothesis_agent.generate(state["idea"], kg, citations)
    hypotheses = orchestrator.rank(hypotheses, kg)

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"Merged knowledge graph: {kg.graph.number_of_nodes()} concepts, "
        f"{kg.graph.number_of_edges()} relations from {len(citations)} citations."
    )
    trace.append(f"Generated and ranked {len(hypotheses)} competing hypotheses.")

    provenance = ResearchProvenance(
        idea=state["idea"],
        run_id=run_id,
        domains=[d["domain"] for d in state.get("domains", [])],
        sources_queried=state.get("sources_queried", []),
        reasoning_trace=trace,
    )

    output = Layer1Output(
        run_id=run_id,
        idea=state["idea"],
        concepts=kg.concepts(),
        relations=kg.relations(),
        citations=citations,
        hypotheses=hypotheses,
        kg_metadata=kg.metadata(),
        provenance=provenance,
    )

    store = ArtifactStore(run_id)
    paths = store.save_layer1(output, kg)
    logger.info("persisted Layer 1 artifacts to %s", store.dir)

    top = output.top(5)
    return {
        "layer1_output": output.model_dump(),
        "top_hypotheses": [h.model_dump() for h in top],
        "reasoning_trace": [f"Persisted {len(paths)} artifacts to {store.dir}."],
    }
