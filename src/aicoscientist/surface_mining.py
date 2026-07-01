"""Deterministic surface-chemistry miner (ADR-001 / ADR-005).

Extracts quantitative AS-ALD knowledge -- inhibitor adsorption energies, reaction
barriers, selectivity percentages, and chemisorb/physisorb regimes -- straight out of the
Layer-1 citation abstracts, so the co-scientist populates its own candidate library from
the literature instead of relying on hand-entered numbers. Runs with no LLM (pure regex),
so ``--offline`` still auto-populates.

Two products:

* :func:`mine_concepts` -- typed ``Inhibitor`` / ``Mechanism`` KG nodes (with the dE value
  in the description) plus ``chemisorbs_on`` / ``physisorbs_on`` relations, so the mined
  knowledge lives in the knowledge graph and is visible/auditable.
* :func:`mine_candidates` -- an ``{inhibitor: priors}`` dict the selection agent reads as
  its default (literature-grounded) priors, each tagged with the source citations and
  whether the non-growth-surface number is extrapolated from a different material.
"""

from __future__ import annotations

import re

from .models import Citation, Concept, Relation, slugify

# Inhibitor vocabulary; extend freely (matching is substring, case-insensitive).
INHIBITOR_VOCAB = [
    "methanesulfonic acid", "octadecylphosphonic acid", "phosphonic acid",
    "ethylbutyric acid", "pivalic acid", "acetic acid", "carboxylic acid",
    "trimethoxypropylsilane", "alkoxysilane", "aniline", "pyrrole", "pyridine",
    "dmatms",
]

# Surfaces we may encounter near an energy; used to tag which surface a number is for.
_NGS_MATERIALS = {"sin": "SiN", "sinx": "SiN", "nitride": "SiN",
                  "ru": "Ru", "co": "Co", "cu": "Cu", "zno": "ZnO", "metal": "metal"}
_GS_MATERIALS = {"sio2": "SiO2", "silica": "SiO2", "oxide": "SiO2", "tio2": "TiO2"}

# Our target non-growth surface for extrapolation flagging.
_TARGET_NGS = "SiN"

_EV = re.compile(r"(-?\d+\.\d+|-?\d+)\s*ev", re.IGNORECASE)
_PCT_SEL = re.compile(r"(\d+\.?\d*)\s*%\s*(?:selectiv|\w{0,12}selectiv)", re.IGNORECASE)
_NM = re.compile(r"(\d+\.?\d*)\s*nm", re.IGNORECASE)


def _nearby(text: str, idx: int, window: int = 48) -> str:
    return text[max(0, idx - window): idx + window].lower()


def _surface_near(ctx: str) -> str | None:
    for token, mat in {**_NGS_MATERIALS, **_GS_MATERIALS}.items():
        if token in ctx:
            return mat
    return None


def mine_from_text(text: str) -> list[dict]:
    """Return one record per inhibitor mentioned in ``text`` with any numbers found."""
    low = text.lower()
    hits: list[dict] = []
    for name in INHIBITOR_VOCAB:
        if name not in low:
            continue
        chem_vals: list[tuple[float, str | None]] = []
        phys_vals: list[tuple[float, str | None]] = []
        for m in _EV.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            # Only exothermic (negative) energies are usable adsorption priors; a positive
            # value denotes an endothermic (unfavorable) path -> not a binding energy.
            if val >= 0:
                continue
            ctx = _nearby(text, m.start())
            surf = _surface_near(ctx)
            if "chemisorb" in ctx or "chemisorption" in ctx:
                chem_vals.append((val, surf))
            elif "physisorb" in ctx or "physisorption" in ctx:
                phys_vals.append((val, surf))
        sel = _PCT_SEL.search(text)
        nm = _NM.search(text)
        hits.append({
            "inhibitor": "acetic acid" if name == "carboxylic acid" else name,
            "chem_vals": chem_vals,
            "phys_vals": phys_vals,
            "selectivity_pct": float(sel.group(1)) if sel else None,
            "thickness_nm": float(nm.group(1)) if nm else None,
            "has_chemisorb": "chemisorb" in low,
            "has_physisorb": "physisorb" in low,
        })
    return hits


def mine_candidates(citations: list[Citation]) -> dict[str, dict]:
    """Aggregate literature-grounded priors per inhibitor across all citations."""
    agg: dict[str, dict] = {}
    for c in citations:
        text = f"{c.title}. {c.abstract or ''}"
        for rec in mine_from_text(text):
            name = rec["inhibitor"]
            entry = agg.setdefault(name, {
                "chem": [], "phys": [], "source_ids": set(),
                "selectivity_pct": None, "thickness_nm": None,
            })
            entry["chem"].extend(rec["chem_vals"])
            entry["phys"].extend(rec["phys_vals"])
            entry["source_ids"].add(c.id)
            if rec["selectivity_pct"] is not None:
                entry["selectivity_pct"] = rec["selectivity_pct"]
            if rec["thickness_nm"] is not None:
                entry["thickness_nm"] = rec["thickness_nm"]

    out: dict[str, dict] = {}
    for name, e in agg.items():
        chem = e["chem"]
        phys = e["phys"]
        # NGS chemisorption: strongest (most negative) reported chemisorption energy.
        dE_ngs = min((v for v, _ in chem), default=None)
        chem_surf = None
        if chem:
            dE_ngs, chem_surf = min(chem, key=lambda vs: vs[0])
        # GS physisorption: weakest binding (closest to zero) physisorption energy.
        dE_gs = max((v for v, _ in phys), default=None)
        extrapolated = bool(chem_surf and chem_surf != _TARGET_NGS)
        if dE_ngs is None and dE_gs is None:
            # Regime known but no number -> still a literature candidate; leave priors None.
            if not (e["chem"] or e["phys"]):
                continue
        out[name] = {
            "dE_ngs": round(dE_ngs, 4) if dE_ngs is not None else None,
            "dE_gs": round(dE_gs, 4) if dE_gs is not None else None,
            "chem_surface": chem_surf,
            "ngs_extrapolated": extrapolated,
            "selectivity_pct": e["selectivity_pct"],
            "thickness_nm": e["thickness_nm"],
            "source_ids": sorted(e["source_ids"]),
            "provenance": "kg-mined",
        }
    return out


def mine_concepts(
    citations: list[Citation], domain: str = "inhibitors"
) -> tuple[list[Concept], list[Relation]]:
    """Typed Inhibitor / Mechanism KG nodes + relations from mined numbers."""
    concepts: list[Concept] = []
    relations: list[Relation] = []
    seen_concepts: set[str] = set()

    def add_concept(name: str, ctype: str, desc: str, srcs: list[str]) -> str:
        cid = slugify(name)
        if cid not in seen_concepts:
            concepts.append(Concept.new(name, type=ctype, description=desc,
                                        domains=[domain], source_ids=srcs))
            seen_concepts.add(cid)
        return cid

    for c in citations:
        text = f"{c.title}. {c.abstract or ''}"
        for rec in mine_from_text(text):
            inh = rec["inhibitor"]
            inh_id = add_concept(inh, "inhibitor", f"AS-ALD inhibitor cited in {c.id}.", [c.id])
            for val, surf in rec["chem_vals"]:
                surf_label = surf or "non-growth surface"
                surf_id = add_concept(surf_label, "surface",
                                      f"Surface referenced in {c.id}.", [c.id])
                mech = f"{inh} chemisorption on {surf_label} (dE={val} eV)"
                mech_id = add_concept(mech, "mechanism",
                                      f"Chemisorption energy {val} eV from {c.id}.", [c.id])
                relations.append(Relation(source_id=inh_id, target_id=surf_id,
                                          relation="chemisorbs_on",
                                          description=f"dE={val} eV", source_ids=[c.id]))
                relations.append(Relation(source_id=inh_id, target_id=mech_id,
                                          relation="measured_by", source_ids=[c.id]))
            for val, surf in rec["phys_vals"]:
                surf_label = surf or "growth surface"
                surf_id = add_concept(surf_label, "surface",
                                      f"Surface referenced in {c.id}.", [c.id])
                relations.append(Relation(source_id=inh_id, target_id=surf_id,
                                          relation="physisorbs_on",
                                          description=f"dE={val} eV", source_ids=[c.id]))
    return concepts, relations
