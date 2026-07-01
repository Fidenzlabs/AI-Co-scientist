"""Deterministic surface-chemistry miner (ADR-001 / ADR-005).

Extracts quantitative AS-ALD knowledge -- inhibitor adsorption energies, reaction
barriers, activation energies (Ea), per-site-type reactivity, selectivity percentages,
and chemisorb/physisorb regimes -- straight out of the Layer-1 citation abstracts, so the
co-scientist populates its own candidate library from the literature instead of relying on
hand-entered numbers. Runs with no LLM (pure regex), so ``--offline`` still auto-populates.

Two products:

* :func:`mine_concepts` -- typed ``Inhibitor`` / ``Mechanism`` KG nodes (with the dE value
  in the description) plus ``chemisorbs_on`` / ``physisorbs_on`` / ``reacts_at`` relations.
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
    "dmatms", "dimethylamino-trimethylsilane", "dimethylamino trimethylsilane",
    "ethyltrichlorosilane", "ets",
]

# Canonical site types (Kim et al. 2026 nomenclature).
SITE_TYPES = ("OH", "O_bridge", "NH2", "NH_bridge")

# Surfaces we may encounter near an energy; used to tag which surface a number is for.
_NGS_MATERIALS = {"sin": "SiN", "sinx": "SiN", "nitride": "SiN",
                  "ru": "Ru", "co": "Co", "cu": "Cu", "zno": "ZnO", "metal": "metal"}
_GS_MATERIALS = {"sio2": "SiO2", "silica": "SiO2", "oxide": "SiO2", "tio2": "TiO2"}

# Our target non-growth surface for extrapolation flagging.
_TARGET_NGS = "SiN"

_EV = re.compile(r"(-?\d+\.\d+|-?\d+)\s*ev", re.IGNORECASE)
_EA = re.compile(
    r"(?:activation\s+energy|Ea|E_a)\s*(?:=|:|of|~)?\s*(-?\d+\.?\d*)\s*eV",
    re.IGNORECASE,
)
_DENSITY = re.compile(
    r"(-?\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*nm\s*[\-−^]\s*2",
    re.IGNORECASE,
)
_PCT_SEL = re.compile(r"(\d+\.?\d*)\s*%\s*(?:selectiv|\w{0,12}selectiv)", re.IGNORECASE)
_NM = re.compile(r"(\d+\.?\d*)\s*nm", re.IGNORECASE)


def _nearby(text: str, idx: int, window: int = 64) -> str:
    return text[max(0, idx - window): idx + window].lower()


def _surface_near(ctx: str) -> str | None:
    for token, mat in {**_NGS_MATERIALS, **_GS_MATERIALS}.items():
        if token in ctx:
            return mat
    return None


def _site_type_near(ctx: str) -> str | None:
    """Tag a site type from context (Kim et al. 2026 nomenclature)."""
    if "siloxane" in ctx or "-o-" in ctx or "o−" in ctx or "o-bridge" in ctx:
        return "O_bridge"
    if "imide" in ctx or "-nh-" in ctx or "nh−" in ctx or "nh-bridge" in ctx:
        return "NH_bridge"
    if "silanol" in ctx or "-oh" in ctx or " oh " in ctx:
        return "OH"
    if "-nh2" in ctx or "amine" in ctx or " nh2" in ctx:
        return "NH2"
    return None


def _canonical_inhibitor(name: str) -> str:
    aliases = {
        "carboxylic acid": "acetic acid",
        "dimethylamino-trimethylsilane": "dmatms",
        "dimethylamino trimethylsilane": "dmatms",
        "ethyltrichlorosilane": "ets",
    }
    return aliases.get(name, name)


def mine_site_densities(text: str) -> dict[str, float]:
    """Extract per-site-type surface densities (sites/nm^2) from abstract text."""
    low = text.lower()
    out: dict[str, float] = {}
    patterns = [
        ("OH", r"silanol\s+-?oh\s+(-?\d+\.?\d*)\s*nm"),
        ("O_bridge", r"siloxane\s+(?:bridge\s+)?-?o-?\s+(-?\d+\.?\d*)\s*nm"),
        ("NH2", r"amine\s+-?nh2?\s+(-?\d+\.?\d*)\s*nm"),
        ("NH_bridge", r"imide\s+(?:bridge\s+)?-?nh-?\s+(-?\d+\.?\d*)\s*nm"),
        ("OH_cryst", r"c-sio2\s+-?oh\s+(-?\d+\.?\d*)\s*nm"),
        ("NH2_cryst", r"c-si3n4\s+-?nh2?\s+(-?\d+\.?\d*)\s*nm"),
    ]
    for key, pat in patterns:
        m = re.search(pat, low)
        if m:
            out[key] = float(m.group(1))
    # Fallback: scan "X nm^-2" near site-type keywords.
    for m in _DENSITY.finditer(text):
        ctx = _nearby(text, m.start(), 40)
        st = _site_type_near(ctx)
        if st and st not in out:
            out[st] = float(m.group(1))
    return out


def mine_from_text(text: str) -> list[dict]:
    """Return one record per inhibitor mentioned in ``text`` with any numbers found."""
    low = text.lower()
    hits: list[dict] = []
    site_densities = mine_site_densities(text)

    for name in INHIBITOR_VOCAB:
        if name not in low:
            continue
        canonical = _canonical_inhibitor(name)
        chem_vals: list[tuple[float, str | None, str | None]] = []
        phys_vals: list[tuple[float, str | None, str | None]] = []
        ea_vals: list[tuple[float, str | None, str | None]] = []

        for m in _EV.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            ctx = _nearby(text, m.start())
            surf = _surface_near(ctx)
            site = _site_type_near(ctx)
            # Skip positive endothermic deltaEr unless tagged as activation energy.
            if val >= 0 and "endothermic" not in ctx and "delta" not in ctx:
                continue
            if "chemisorb" in ctx or "chemisorption" in ctx or "exothermic" in ctx:
                chem_vals.append((val, surf, site))
            elif "physisorb" in ctx or "physisorption" in ctx:
                phys_vals.append((val, surf, site))
            elif "endothermic" in ctx or "delta" in ctx:
                chem_vals.append((val, surf, site))  # deltaEr (may be positive)

        for m in _EA.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if val <= 0:
                continue
            ctx = _nearby(text, m.start())
            surf = _surface_near(ctx)
            site = _site_type_near(ctx)
            ea_vals.append((val, surf, site))

        sel = _PCT_SEL.search(text)
        nm = _NM.search(text)
        hits.append({
            "inhibitor": canonical,
            "chem_vals": chem_vals,
            "phys_vals": phys_vals,
            "ea_vals": ea_vals,
            "site_densities": site_densities,
            "selectivity_pct": float(sel.group(1)) if sel else None,
            "thickness_nm": float(nm.group(1)) if nm else None,
            "has_chemisorb": "chemisorb" in low,
            "has_physisorb": "physisorb" in low,
        })
    return hits


def _site_reactivity(rec: dict) -> dict[str, dict]:
    """Build per-site-type reactivity priors from a mined record."""
    sites: dict[str, dict] = {}
    for val, surf, site in rec.get("chem_vals", []):
        if site:
            sites.setdefault(site, {})["deltaEr_eV"] = val
            if surf:
                sites[site]["surface"] = surf
    for val, surf, site in rec.get("ea_vals", []):
        if site:
            sites.setdefault(site, {})["Ea_eV"] = val
            if surf:
                sites[site]["surface"] = surf
    return sites


def mine_candidates(citations: list[Citation]) -> dict[str, dict]:
    """Aggregate literature-grounded priors per inhibitor across all citations."""
    agg: dict[str, dict] = {}
    for c in citations:
        text = f"{c.title}. {c.abstract or ''}"
        for rec in mine_from_text(text):
            name = rec["inhibitor"]
            entry = agg.setdefault(name, {
                "chem": [], "phys": [], "ea": [], "source_ids": set(),
                "selectivity_pct": None, "thickness_nm": None,
                "site_reactivity": {}, "site_densities": {},
            })
            entry["chem"].extend(rec["chem_vals"])
            entry["phys"].extend(rec["phys_vals"])
            entry["ea"].extend(rec["ea_vals"])
            entry["source_ids"].add(c.id)
            entry["site_reactivity"].update(_site_reactivity(rec))
            entry["site_densities"].update(rec.get("site_densities", {}))
            if rec["selectivity_pct"] is not None:
                entry["selectivity_pct"] = rec["selectivity_pct"]
            if rec["thickness_nm"] is not None:
                entry["thickness_nm"] = rec["thickness_nm"]

    out: dict[str, dict] = {}
    for name, e in agg.items():
        chem = e["chem"]
        phys = e["phys"]
        dE_ngs = min((v for v, _, _ in chem), default=None)
        chem_surf = None
        if chem:
            dE_ngs, chem_surf, _ = min(chem, key=lambda vs: vs[0])
        dE_gs = max((v for v, _, _ in phys), default=None)
        extrapolated = bool(chem_surf and chem_surf != _TARGET_NGS)
        if dE_ngs is None and dE_gs is None and not e["site_reactivity"]:
            if not (e["chem"] or e["phys"] or e["ea"]):
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
            "site_reactivity": e["site_reactivity"],
            "site_densities": e["site_densities"],
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
            for val, surf, site in rec["chem_vals"]:
                surf_label = surf or "non-growth surface"
                surf_id = add_concept(surf_label, "surface",
                                      f"Surface referenced in {c.id}.", [c.id])
                site_label = site or "terminal"
                site_id = add_concept(
                    f"{surf_label} {site_label} site", "surface_site",
                    f"Reactive site type from {c.id}.", [c.id],
                )
                mech = f"{inh} chemisorption on {surf_label} {site_label} (dE={val} eV)"
                mech_id = add_concept(mech, "mechanism",
                                      f"Chemisorption energy {val} eV from {c.id}.", [c.id])
                relations.append(Relation(source_id=inh_id, target_id=surf_id,
                                          relation="chemisorbs_on",
                                          description=f"dE={val} eV", source_ids=[c.id]))
                relations.append(Relation(source_id=inh_id, target_id=site_id,
                                          relation="reacts_at",
                                          description=f"dE={val} eV at {site_label}",
                                          source_ids=[c.id]))
                relations.append(Relation(source_id=inh_id, target_id=mech_id,
                                          relation="measured_by", source_ids=[c.id]))
            for val, surf, site in rec["phys_vals"]:
                surf_label = surf or "growth surface"
                surf_id = add_concept(surf_label, "surface",
                                      f"Surface referenced in {c.id}.", [c.id])
                relations.append(Relation(source_id=inh_id, target_id=surf_id,
                                          relation="physisorbs_on",
                                          description=f"dE={val} eV", source_ids=[c.id]))
            for val, surf, site in rec.get("ea_vals", []):
                site_label = site or "terminal"
                site_id = add_concept(
                    f"{surf or 'surface'} {site_label} site", "surface_site",
                    f"Reactive site type from {c.id}.", [c.id],
                )
                mech = f"{inh} activation at {site_label} (Ea={val} eV)"
                mech_id = add_concept(mech, "mechanism",
                                      f"Activation energy {val} eV from {c.id}.", [c.id])
                relations.append(Relation(source_id=inh_id, target_id=site_id,
                                          relation="reacts_at",
                                          description=f"Ea={val} eV at {site_label}",
                                          source_ids=[c.id]))
                relations.append(Relation(source_id=inh_id, target_id=mech_id,
                                          relation="measured_by", source_ids=[c.id]))
    return concepts, relations
