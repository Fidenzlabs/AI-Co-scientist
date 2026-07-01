"""Layer 2 — Human-in-the-Loop Scientific Reasoning.

The pipeline pauses on a LangGraph ``interrupt`` that surfaces the five highest-ranked
hypotheses (with supporting/contradicting evidence, confidence, citations, reasoning
trace, and novelty). The researcher's structured decision resumes the graph, and
``apply_decision`` turns it into the official research hypothesis.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from .asald import derive_asald_spec
from .layer1_graph import CoScientistState
from .models import (
    Evidence,
    Hypothesis,
    HypothesisStateGraph,
    OfficialHypothesis,
    ResearcherDecision,
)
from .persistence import ArtifactStore

logger = logging.getLogger(__name__)


def present_hypotheses(state: CoScientistState) -> dict:
    """Pause and present the top-5 hypotheses to the researcher."""
    top = state.get("top_hypotheses", [])
    decision = interrupt(
        {
            "type": "hypothesis_review",
            "idea": state.get("idea"),
            "instructions": (
                "Review the top hypotheses and choose an action: "
                "select / modify / merge / new / quit."
            ),
            "hypotheses": top,
        }
    )
    if not isinstance(decision, dict):
        decision = {"action": "quit", "notes": "invalid decision payload"}
    return {"decision": decision}


def apply_decision(state: CoScientistState) -> dict:
    """Translate the researcher's decision into the official hypothesis."""
    decision = ResearcherDecision.model_validate(state.get("decision") or {"action": "quit"})
    run_id = state["run_id"]
    hyp_by_id = {
        h["id"]: Hypothesis.model_validate(h)
        for h in state.get("layer1_output", {}).get("hypotheses", [])
    }

    if decision.action == "quit":
        logger.info("researcher aborted without selecting a hypothesis")
        return {"official_hypothesis": None}

    selected = [hyp_by_id[i] for i in decision.selected_ids if i in hyp_by_id]

    if decision.action == "select" and selected:
        base = selected[0]
        official = OfficialHypothesis(
            run_id=run_id,
            statement=base.statement,
            origin=decision,
            state_graph=base.state_graph,
            source_hypothesis_ids=[base.id],
        )
    elif decision.action == "modify" and selected:
        base = selected[0]
        official = OfficialHypothesis(
            run_id=run_id,
            statement=decision.statement or base.statement,
            origin=decision,
            state_graph=base.state_graph,
            source_hypothesis_ids=[base.id],
        )
    elif decision.action == "merge" and selected:
        official = OfficialHypothesis(
            run_id=run_id,
            statement=decision.statement or _merged_statement(selected),
            origin=decision,
            state_graph=_merge_state_graphs(selected),
            source_hypothesis_ids=[h.id for h in selected],
        )
    else:  # "new" or a degenerate selection -> fresh direction
        official = OfficialHypothesis(
            run_id=run_id,
            statement=decision.statement or "New research direction (unspecified).",
            origin=decision,
            state_graph=HypothesisStateGraph(
                assumptions=["Researcher-introduced direction; evidence to be gathered."],
                confidence=0.5,
            ),
            source_hypothesis_ids=[h.id for h in selected],
        )

    # Derive the structured AS-ALD intervention (GS/NGS/inhibitor/precursor/target)
    # that Layer 3's surface builder + reactivity engine consume.
    concept_names = [
        c.get("name", "")
        for c in state.get("layer1_output", {}).get("concepts", [])
    ]
    official.asald = derive_asald_spec(
        official.statement,
        concept_names=concept_names,
        provenance_refs=official.state_graph.references,
    )

    ArtifactStore(run_id).save_official_hypothesis(official)
    logger.info("saved official hypothesis for run %s", run_id)
    return {"official_hypothesis": official.model_dump()}


def _merged_statement(hyps: list[Hypothesis]) -> str:
    return " AND ".join(h.statement.rstrip(".") for h in hyps) + "."


def _merge_state_graphs(hyps: list[Hypothesis]) -> HypothesisStateGraph:
    support: list[Evidence] = []
    contra: list[Evidence] = []
    assumptions: list[str] = []
    related: list[str] = []
    refs: list[str] = []
    for h in hyps:
        sg = h.state_graph
        support.extend(sg.supporting_evidence)
        contra.extend(sg.contradicting_evidence)
        assumptions.extend(sg.assumptions)
        related.extend(sg.related_concept_ids)
        refs.extend(sg.references)
    confidence = sum(h.state_graph.confidence for h in hyps) / max(1, len(hyps))
    return HypothesisStateGraph(
        supporting_evidence=support,
        contradicting_evidence=contra,
        assumptions=_unique(assumptions),
        related_concept_ids=_unique(related),
        confidence=round(confidence, 3),
        references=_unique(refs),
    )


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
