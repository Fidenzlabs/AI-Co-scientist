"""Per-run artifact store.

Writes the six persistent Layer 1 artifact types plus the Layer 2 official hypothesis
under ``artifacts/<run_id>/``. Everything is JSON (plus a GraphML copy of the KG) so
artifacts are portable and become the input to later layers.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import get_settings
from .knowledge_graph import KnowledgeGraph
from .models import Layer1Output, Layer3Output, OfficialHypothesis


class ArtifactStore:
    def __init__(self, run_id: str) -> None:
        settings = get_settings()
        self.run_id = run_id
        self.dir = settings.artifacts_path / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def datasets_dir(self) -> Path:
        path = self.dir / "datasets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        path = self.dir / "simulation_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, name: str, payload) -> Path:
        path = self.dir / name
        if hasattr(payload, "model_dump"):
            data = payload.model_dump()
        else:
            data = payload
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def save_layer1(self, output: Layer1Output, kg: KnowledgeGraph) -> dict[str, Path]:
        paths: dict[str, Path] = {}

        # Knowledge graph (JSON + GraphML)
        kg.save(self.dir / "knowledge_graph.json", self.dir / "knowledge_graph.graphml")
        paths["knowledge_graph"] = self.dir / "knowledge_graph.json"
        paths["knowledge_graph_graphml"] = self.dir / "knowledge_graph.graphml"

        # Knowledge graph metadata
        paths["kg_metadata"] = self._write(
            "knowledge_graph_metadata.json", output.kg_metadata
        )

        # Citation repository
        paths["citations"] = self._write(
            "citation_repository.json",
            {"citations": [c.model_dump() for c in output.citations]},
        )

        # Hypothesis state graphs
        paths["hypotheses"] = self._write(
            "hypothesis_state_graphs.json",
            {"hypotheses": [h.model_dump() for h in output.hypotheses]},
        )

        # Research provenance
        if output.provenance is not None:
            paths["provenance"] = self._write(
                "research_provenance.json", output.provenance
            )

        # Confidence scores (ranking breakdown per hypothesis)
        paths["confidence_scores"] = self._write(
            "confidence_scores.json",
            {
                "scores": [
                    {
                        "id": h.id,
                        "statement": h.statement,
                        **h.scores.model_dump(),
                    }
                    for h in sorted(
                        output.hypotheses,
                        key=lambda h: h.scores.composite,
                        reverse=True,
                    )
                ]
            },
        )

        # Full Layer 1 output for convenient reloading.
        paths["layer1_output"] = self._write("layer1_output.json", output)
        return paths

    def save_official_hypothesis(self, official: OfficialHypothesis) -> Path:
        return self._write("official_hypothesis.json", official)

    def save_layer3(self, output: Layer3Output, kg: KnowledgeGraph) -> dict[str, Path]:
        paths: dict[str, Path] = {}

        paths["validation_plan"] = self._write("validation_plan.json", output.result.plan)

        paths["validation_results"] = self._write(
            "validation_results.json",
            {
                "final": output.result.model_dump(),
                "history": [r.model_dump() for r in output.history],
                "reflections": [r.model_dump() for r in output.reflections],
                "iterations": output.iterations,
                "agentic_pattern": output.agentic_pattern,
                "methodology_citations": output.methodology_citations,
            },
        )

        paths["validation_provenance"] = self._write(
            "validation_provenance.json",
            {
                "run_id": output.run_id,
                "hypothesis": output.hypothesis_statement,
                "agentic_pattern": output.agentic_pattern,
                "methodology_citations": output.methodology_citations,
                "iterations": output.iterations,
                "loop_trace": [
                    {
                        "iteration": r.plan.iteration,
                        "domain": r.plan.domain,
                        "method": r.plan.method,
                        "verdict": r.verdict.value,
                        "confidence": r.confidence,
                        "reflection": (
                            output.reflections[i].decision
                            if i < len(output.reflections)
                            else None
                        ),
                    }
                    for i, r in enumerate(output.history)
                ],
                "generated_at": output.generated_at,
            },
        )

        paths["layer3_output"] = self._write("layer3_output.json", output)

        # Re-serialize the KG now that it contains validation nodes/edges.
        kg.save(self.dir / "knowledge_graph.json", self.dir / "knowledge_graph.graphml")
        paths["knowledge_graph"] = self.dir / "knowledge_graph.json"

        return paths
