"""Agentic inhibitor/precursor selection agent (ADR-005, Deliverable #2).

The Layer-3 ReAct designer, specialized for AS-ALD. It:

1. Retrieves candidate inhibitors/precursors from the Layer-1 knowledge graph.
2. Ranks them against a human-editable ``selection_criteria.md`` (volatility,
   functional-group <-> site compatibility, chemisorb-vs-physisorb differential, sterics,
   removability), grounded in the committed :class:`ASALDSpec`.
3. Encodes the chosen ``(inhibitor, precursor)`` pair and its literature adsorption-energy
   priors into a runnable ``ValidationPlan`` for the ``surface_reactivity`` engine.
4. On a Reflection ``refine`` it advances to the next-ranked candidate pair (the swarm-style
   "keep exploring" behavior), bounded by ``MAX_VALIDATION_ITERS``.

Without an LLM key it uses a deterministic ranking so offline runs are fully reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import get_settings
from ..models import (
    ASALDSpec,
    OfficialHypothesis,
    Reflection,
    SuccessCriterion,
    ValidationPlan,
)

logger = logging.getLogger(__name__)

VALID_DOMAINS = {"surface_reactivity"}

_VOL = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Built-in fallback library if selection_criteria.md is missing/unparseable.
_DEFAULT_LIBRARY = {
    "inhibitors": {
        "acetic acid": {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "pivalic acid": {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "methanesulfonic acid": {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid", "volatility": "medium", "removability": "medium"},
        "aniline": {"dE_ngs": -0.90, "dE_gs": -0.15, "functional_group": "aromatic amine", "volatility": "high", "removability": "high"},
        "octadecylphosphonic acid": {"dE_ngs": -1.30, "dE_gs": -0.30, "functional_group": "phosphonic acid", "volatility": "low", "removability": "low"},
    },
    "precursors": {
        "BDEAS": {"target_film": "SiOx"},
        "DIPAS": {"target_film": "SiOx"},
        "HCDS": {"target_film": "SiOx"},
        "TDMAT": {"target_film": "TiN"},
        "TMA": {"target_film": "Al2O3"},
    },
}


def _load_library(path: str) -> dict:
    """Parse the ```json``` candidate block from selection_criteria.md."""
    p = Path(path)
    if not p.exists():
        logger.warning("selection_criteria.md not found at %s; using default library", path)
        return _DEFAULT_LIBRARY
    text = p.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        logger.warning("no json candidate block in %s; using default library", path)
        return _DEFAULT_LIBRARY
    try:
        lib = json.loads(m.group(1))
        if "inhibitors" in lib and "precursors" in lib:
            return lib
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse candidate block (%s); using default library", exc)
    return _DEFAULT_LIBRARY


class ExperimentDesigner:
    """AS-ALD selection agent (kept named ExperimentDesigner for the Layer-3 graph)."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def design(
        self,
        official: OfficialHypothesis,
        concept_names: list[str] | None = None,
        prior_critique: Reflection | None = None,
        iteration: int = 0,
    ) -> ValidationPlan:
        settings = get_settings()
        concept_names = concept_names or []
        spec = official.asald or ASALDSpec()
        library = _load_library(settings.selection_criteria_path)

        ranked = self._rank_inhibitors(library, spec, concept_names)
        # On refinement, advance to the next-ranked inhibitor (persist / keep exploring).
        idx = min(iteration, len(ranked) - 1)
        inhibitor, props = ranked[idx]

        precursor, target_film = self._choose_precursor(library, spec)

        trace = [
            f"Retrieved {len(ranked)} inhibitor candidates grounded in the KG "
            f"(criteria: {settings.selection_criteria_path}).",
            f"Ranked by differential adsorption + volatility + removability; "
            f"selected '{inhibitor}' (rank {idx + 1}).",
            f"Paired with precursor '{precursor}' for target film {target_film}.",
        ]
        if prior_critique and prior_critique.decision == "refine":
            trace.append(f"Refinement: {prior_critique.critique}")

        plan = ValidationPlan(
            domain="surface_reactivity",
            method=(
                f"AS-ALD differential-reactivity protocol (ADR-009): inhibitor "
                f"adsorption screen on {spec.non_growth_surface} (NGS) vs "
                f"{spec.growth_surface} (GS) over a gated surface ensemble, "
                f"blocking coverage -> nucleation delay -> S(N)."
            ),
            rationale=(
                f"'{inhibitor}' ({props.get('functional_group', 'n/a')}) is predicted to "
                f"chemisorb on the NGS and physisorb on the GS, giving a large differential "
                f"blocking coverage."
            ),
            reasoning_trace=trace,
            assumptions=[
                "Adsorption-energy priors are literature/xTB values; Tier-1 MLIP recomputes "
                "them within a single calculator/head/dtype.",
                "Only chemisorbed, purge-surviving inhibitor blocks the precursor.",
                "Selectivity is reported as mean +/- std over the surface ensemble.",
            ],
            seed=42 + iteration,
            iteration=iteration,
            data_spec={
                "inhibitor": inhibitor,
                "precursor": precursor,
                "growth_surface": spec.growth_surface,
                "non_growth_surface": spec.non_growth_surface,
                "target_film": target_film,
                "target_thickness_nm": spec.target_thickness_nm,
                "target_selectivity": spec.target_selectivity,
                "dE_ngs_eV": props["dE_ngs"],
                "dE_gs_eV": props["dE_gs"],
                "dE_prior_std": 0.08,
                "literature_dE_ngs_eV": props["dE_ngs"],
                "temperature_K": settings.ald_temperature_k,
                "dose_ratio": 1.0,
                "ensemble_n": settings.surface_ensemble_n,
                "compute_tier": settings.compute_tier,
                "provenance_refs": spec.provenance_refs,
            },
            success_criteria=[
                SuccessCriterion(
                    metric="S_at_target", operator=">=", threshold=spec.target_selectivity,
                    description=f"selectivity meets the {spec.target_selectivity:.0%} target",
                ),
                SuccessCriterion(
                    metric="differential_blocking", operator=">=", threshold=0.5,
                    description="NGS is blocked substantially more than GS",
                ),
                SuccessCriterion(
                    metric="dE_ngs_mean_eV", operator="<", threshold=-0.7,
                    description="chemisorption on the non-growth surface",
                ),
            ],
        )
        return plan

    # ──────────────────────── ranking ────────────────────────

    def _rank_inhibitors(
        self, library: dict, spec: ASALDSpec, concept_names: list[str]
    ) -> list[tuple[str, dict]]:
        inhibitors = library.get("inhibitors", {})
        kg_text = " ".join(concept_names).lower()

        def score(name: str, props: dict) -> float:
            differential = props["dE_gs"] - props["dE_ngs"]     # want large positive
            vol = _VOL.get(str(props.get("volatility", "medium")).lower(), 0.5)
            rem = _VOL.get(str(props.get("removability", "medium")).lower(), 0.5)
            s = differential + 0.2 * vol + 0.2 * rem
            if name.lower() in kg_text:      # grounded in the literature KG -> preferred
                s += 0.3
            if name.lower() == spec.inhibitor.lower():  # honor the committed choice first
                s += 1.0
            return s

        ranked = sorted(inhibitors.items(), key=lambda kv: score(*kv), reverse=True)
        if not ranked:  # ensure at least the committed inhibitor is runnable
            ranked = [(spec.inhibitor, {"dE_ngs": -1.0, "dE_gs": -0.2,
                                        "functional_group": "n/a"})]
        return ranked

    @staticmethod
    def _choose_precursor(library: dict, spec: ASALDSpec) -> tuple[str, str]:
        precursors = library.get("precursors", {})
        if spec.precursor in precursors:
            return spec.precursor, precursors[spec.precursor].get("target_film", spec.target_film)
        # else pick a precursor matching the target film, defaulting to the committed one.
        for name, props in precursors.items():
            if props.get("target_film", "").lower() == spec.target_film.lower():
                return name, props["target_film"]
        return spec.precursor, spec.target_film
