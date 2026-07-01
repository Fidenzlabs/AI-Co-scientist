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

# Built-in fallback library, literature-informed. NGS numbers for a-SiN are estimates
# (published DFT uses metal/oxide NGS, e.g. aniline chemisorbs -3.59 eV on Ru, -2.17 eV
# on Co); GS (SiO2) physisorption numbers are taken from the cited DFT where available
# (aniline -0.57 eV, Langmuir 2023 10.1021/acs.langmuir.2c03214).
_DEFAULT_LIBRARY = {
    "inhibitors": {
        "acetic acid": {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "pivalic acid": {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "ethylbutyric acid": {"dE_ngs": -0.98, "dE_gs": -0.24, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "methanesulfonic acid": {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid", "volatility": "medium", "removability": "medium"},
        "aniline": {"dE_ngs": -0.90, "dE_gs": -0.57, "functional_group": "aromatic amine", "volatility": "high", "removability": "high"},
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


_MERGE_KEYS = ("dE_ngs", "dE_gs", "functional_group", "volatility", "removability")


def _load_manual_library(path: str) -> dict:
    """Parse the ```json``` candidate block from selection_criteria.md (human override)."""
    p = Path(path)
    if not p.exists():
        logger.info("selection_criteria.md not found at %s; no manual overrides", path)
        return {}
    text = p.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        logger.info("no json candidate block in %s; no manual overrides", path)
        return {}
    try:
        lib = json.loads(m.group(1))
        if isinstance(lib, dict):
            return lib
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse candidate block (%s); ignoring manual overrides", exc)
    return {}


def _load_kg_candidates(run_id: str) -> dict:
    """Load the literature-mined candidate library written by Layer 1 (kg_candidates.json)."""
    settings = get_settings()
    p = Path(settings.artifacts_path) / run_id / "kg_candidates.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("inhibitors", {}) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read kg_candidates.json (%s)", exc)
        return {}


def _merge_libraries(manual: dict, kg: dict, run_id: str) -> tuple[dict, dict]:
    """Merge inhibitor priors with precedence set by PRIORS_SOURCE.

    Base is the built-in default; KG-mined and manual layers overlay it. In ``auto`` mode
    KG-mined values win; in ``manual`` mode selection_criteria.md wins. Returns the merged
    ``{inhibitors, precursors}`` library plus a ``provenance`` map recording, per inhibitor,
    which layer set ``dE_ngs`` and whether that value is extrapolated from another surface.
    """
    settings = get_settings()
    manual_inh = (manual or {}).get("inhibitors", {})
    default_inh = _DEFAULT_LIBRARY["inhibitors"]

    # Precedence: later entries override earlier ones for the same key.
    if settings.priors_source.lower() == "manual":
        layers = [("builtin", default_inh), ("kg-mined", kg), ("manual", manual_inh)]
    else:  # auto (default): literature (KG) wins over the shipped manual defaults
        layers = [("builtin", default_inh), ("manual", manual_inh), ("kg-mined", kg)]

    names = set().union(*(set(layer.keys()) for _, layer in layers))
    merged: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for name in names:
        entry: dict = {}
        prov = {"dE_ngs_source": "builtin", "ngs_extrapolated": False, "source_ids": []}
        for src_name, layer in layers:
            src = layer.get(name)
            if not src:
                continue
            for k in _MERGE_KEYS:
                v = src.get(k)
                if v is not None:
                    entry[k] = v
                    if k == "dE_ngs":
                        prov["dE_ngs_source"] = src_name
                        prov["ngs_extrapolated"] = bool(src.get("ngs_extrapolated", False))
                        prov["source_ids"] = src.get("source_ids", prov["source_ids"])
        entry.setdefault("dE_ngs", -1.0)
        entry.setdefault("dE_gs", -0.2)
        merged[name] = entry
        provenance[name] = prov

    precursors = (manual or {}).get("precursors") or _DEFAULT_LIBRARY["precursors"]
    return {"inhibitors": merged, "precursors": precursors}, provenance


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

        manual = _load_manual_library(settings.selection_criteria_path)
        kg = _load_kg_candidates(official.run_id)
        library, provenance = _merge_libraries(manual, kg, official.run_id)

        ranked = self._rank_inhibitors(library, spec, concept_names)
        # On refinement, advance to the next-ranked inhibitor (persist / keep exploring).
        idx = min(iteration, len(ranked) - 1)
        inhibitor, props = ranked[idx]
        prov = provenance.get(inhibitor, {"dE_ngs_source": "builtin",
                                          "ngs_extrapolated": False, "source_ids": []})

        precursor, target_film = self._choose_precursor(library, spec)

        n_kg = len(kg)
        n_manual = len((manual or {}).get("inhibitors", {}))
        trace = [
            f"Prior sources merged (mode={settings.priors_source}): {n_kg} KG-mined, "
            f"{n_manual} manual, {len(_DEFAULT_LIBRARY['inhibitors'])} built-in defaults.",
            f"Retrieved {len(ranked)} inhibitor candidates; ranked by differential "
            f"adsorption + volatility + removability.",
            f"Selected '{inhibitor}' (rank {idx + 1}); dE_ngs from '{prov['dE_ngs_source']}'"
            + (" [extrapolated from another NGS material]" if prov["ngs_extrapolated"] else "")
            + ".",
            f"Paired with precursor '{precursor}' for target film {target_film}.",
        ]
        if prov["source_ids"]:
            trace.append(f"dE_ngs supported by citations: {', '.join(prov['source_ids'])}.")
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
                "prior_source": prov["dE_ngs_source"],
                "prior_extrapolated": prov["ngs_extrapolated"],
                "prior_source_ids": prov["source_ids"],
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
                s += 100.0
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
