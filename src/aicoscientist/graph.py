"""Compose Layers 1 and 2 into a single checkpointed LangGraph.

    plan_domains -> [run_domain ...] -> synthesize -> present_hypotheses (interrupt)
                 -> apply_decision -> END

A ``SqliteSaver`` checkpointer is required so the graph can pause at the Layer 2
``interrupt`` and be resumed later with the researcher's decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .layer1_graph import (
    CoScientistState,
    fan_out_domains,
    plan_domains,
    run_domain,
    synthesize,
)
from .layer2_graph import apply_decision, present_hypotheses


def build_graph_builder() -> StateGraph:
    builder = StateGraph(CoScientistState)

    builder.add_node("plan_domains", plan_domains)
    builder.add_node("run_domain", run_domain)
    builder.add_node("synthesize", synthesize)
    builder.add_node("present_hypotheses", present_hypotheses)
    builder.add_node("apply_decision", apply_decision)

    builder.add_edge(START, "plan_domains")
    builder.add_conditional_edges("plan_domains", fan_out_domains, ["run_domain"])
    builder.add_edge("run_domain", "synthesize")
    builder.add_edge("synthesize", "present_hypotheses")
    builder.add_edge("present_hypotheses", "apply_decision")
    builder.add_edge("apply_decision", END)

    return builder


@contextmanager
def compiled_graph(checkpoint_path: str | None = None):
    """Yield a compiled graph with a SqliteSaver checkpointer.

    Used as a context manager so the underlying SQLite connection is closed cleanly.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = get_settings()
    db_path = checkpoint_path or str(settings.artifacts_path / "checkpoints.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield build_graph_builder().compile(checkpointer=checkpointer)
