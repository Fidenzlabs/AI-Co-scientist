"""Derive a structured :class:`ASALDSpec` from a committed hypothesis statement.

Layer 2 commits a free-text intervention hypothesis; Layer 3 needs the structured
(growth surface, non-growth surface, inhibitor, precursor, target film, thickness,
selectivity) tuple. This module maps the statement (plus KG concepts and provenance
DOIs) onto that tuple using surface-chemistry vocabulary, falling back to the ADR's
verified worked example so the pipeline always has a runnable spec.
"""

from __future__ import annotations

import re

from .models import ASALDSpec

# Known AS-ALD vocabulary (literature-grounded candidate library).
_INHIBITORS = [
    "methanesulfonic acid", "octadecylphosphonic acid", "phosphonic acid",
    "pivalic acid", "ethylbutyric acid", "acetic acid", "carboxylic acid",
    "aniline", "dmatms", "trimethylsilyl", "silyl",
]
_PRECURSORS = ["bdeas", "dipas", "hcds", "tdmat", "dmai", "tma"]

# Growth / non-growth surface synonyms.
_GROWTH = ["a-sio2", "sio2", "silica", "silicon oxide", "oxide growth"]
_NONGROWTH = ["a-sin", "sin", "sinx", "silicon nitride", "nitride", "-nh"]

_FILMS = {
    "siox": "SiOx", "sio2": "SiOx", "al2o3": "Al2O3", "tin": "TiN",
    "zro2": "ZrO2", "tio2": "TiO2", "hfo2": "HfO2",
}


def _canonical_inhibitor(text: str) -> str | None:
    for name in _INHIBITORS:
        if name in text:
            return "acetic acid" if name == "carboxylic acid" else name
    return None


def _canonical_precursor(text: str) -> str | None:
    for name in _PRECURSORS:
        if name in text:
            return name.upper()
    return None


def _target_thickness_nm(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:angstrom|å|a)\b", text)
    if m:
        return float(m.group(1)) / 10.0
    return None


def _target_selectivity(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"selectivit\w*\s*(?:of|>=|>|=|~)?\s*(0?\.\d+)", text)
    if m:
        return float(m.group(1))
    return None


def _target_film(text: str) -> str | None:
    for key, val in _FILMS.items():
        if key in text:
            return val
    return None


def derive_asald_spec(
    statement: str,
    concept_names: list[str] | None = None,
    provenance_refs: list[str] | None = None,
) -> ASALDSpec:
    """Best-effort structured AS-ALD spec, defaulting to the verified worked example."""
    corpus = (statement + " " + " ".join(concept_names or [])).lower()
    defaults = ASALDSpec()  # the ADR worked example (acetic acid / BDEAS, a-SiO2/a-SiN)

    inhibitor = _canonical_inhibitor(corpus) or defaults.inhibitor
    precursor = _canonical_precursor(corpus) or defaults.precursor

    growth = defaults.growth_surface
    non_growth = defaults.non_growth_surface
    if any(g in corpus for g in _GROWTH):
        growth = "a-SiO2"
    if any(n in corpus for n in _NONGROWTH):
        non_growth = "a-SiN"

    return ASALDSpec(
        growth_surface=growth,
        non_growth_surface=non_growth,
        inhibitor=inhibitor,
        precursor=precursor,
        target_film=_target_film(corpus) or defaults.target_film,
        target_thickness_nm=_target_thickness_nm(corpus) or defaults.target_thickness_nm,
        target_selectivity=_target_selectivity(corpus) or defaults.target_selectivity,
        provenance_refs=list(provenance_refs or []),
    )
