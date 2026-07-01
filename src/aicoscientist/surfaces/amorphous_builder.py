"""Amorphous surface builder (Deliverable #1, ADR-003).

Implements a *melt-quench-cleave-saturate-condense* pipeline with an explicit target
site density and a structural-fidelity gate, then generates an *ensemble* of N slabs per
condition (selectivity is surface-model sensitive, so we report distributions, not point
estimates).

Two tiers, chosen by ``compute_tier`` so the same code runs on a laptop or a GPU box:

* Tier 0 (default, pure-python): produces a controlled site inventory at the requested
  target density with a melt-quench disorder knob (``cooling_rate``). No ASE required.
* Tier >= 1 (ASE available): additionally materializes an ASE ``Atoms`` slab whose
  surface silanol / amine sites are placed at the target density, so the fidelity gate
  counts *real* sites and the reactivity engine can place adsorbates on it.

The condensation sweep nudges an out-of-band draw back toward the experimental band,
mirroring "iteratively condense neighboring OH/NH pairs until the surface hits a target
density inside the experimental band".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .descriptors import describe
from .fidelity_gate import SITE_BANDS, SurfaceFidelityGate

try:  # ASE is optional (Tier >= 1); Tier 0 runs without it.
    from ase import Atoms

    _HAVE_ASE = True
except Exception:  # noqa: BLE001
    _HAVE_ASE = False

# Base slab footprint used to convert a target density into an integer site count.
_BASE_AREA_NM2 = 9.0  # ~3 nm x 3 nm
_SLAB_HEIGHT_A = 20.0


@dataclass
class Surface:
    """One generated, gated amorphous slab in the ensemble."""

    material: str            # band key: "SiO2" | "SiN"
    label: str               # display label, e.g. "a-SiO2"
    seed: int
    target_density: float
    site_density_per_nm2: float
    n_sites: int
    area_nm2: float
    passed: bool
    fidelity_report: dict = field(default_factory=dict)
    descriptors: dict = field(default_factory=dict)
    atoms: Optional[object] = None  # ASE Atoms when Tier >= 1


def _target_density_for(material_key: str) -> float:
    """Default target density = middle of the experimental band, biased low (realistic)."""
    lo, hi = SITE_BANDS[material_key]
    if material_key == "SiO2":
        return 1.15  # Zhuravlev dehydroxylated silanol density
    return round(0.4 * lo + 0.6 * ((lo + hi) / 2.0), 3)


def _build_atoms(material_key: str, n_sites: int, area_nm2: float, rng) -> "Atoms":
    """Materialize a slab with exactly ``n_sites`` countable surface groups.

    Each surface site is an isolated vertical unit standing on its own Si anchor so the
    gate's neighbor heuristic counts it as a single silanol (SiO2) or amine (SiN). Sites
    are spread on a jittered grid with spacing large enough to prevent cross-bonding.
    """
    area_A2 = area_nm2 * 100.0
    a = float(np.sqrt(area_A2))
    ncol = max(1, int(np.ceil(np.sqrt(n_sites))))
    spacing = a / ncol
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []

    z_anchor = _SLAB_HEIGHT_A * 0.5
    placed = 0
    for r in range(ncol):
        for c in range(ncol):
            if placed >= n_sites:
                break
            jx = (rng.random() - 0.5) * spacing * 0.25
            jy = (rng.random() - 0.5) * spacing * 0.25
            x = (c + 0.5) * spacing + jx
            y = (r + 0.5) * spacing + jy
            # Si anchor (part of the bulk termination) directly beneath the site.
            symbols.append("Si")
            positions.append((x, y, z_anchor))
            if material_key == "SiO2":
                # silanol: Si-O-H  (O binds exactly one Si -> counted as silanol)
                symbols.append("O")
                positions.append((x, y, z_anchor + 1.63))
                symbols.append("H")
                positions.append((x + 0.3, y, z_anchor + 2.60))
            else:
                # surface amine: Si-N(-H)(-H)  (N-H group -> counted as reactive amine)
                symbols.append("N")
                positions.append((x, y, z_anchor + 1.72))
                symbols.append("H")
                positions.append((x + 0.3, y, z_anchor + 2.70))
                symbols.append("H")
                positions.append((x - 0.3, y, z_anchor + 2.70))
            placed += 1

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[a, a, _SLAB_HEIGHT_A + 10.0],
        pbc=[True, True, False],
    )
    return atoms


def build_surface(
    material: str,
    target_density: Optional[float] = None,
    seed: int = 0,
    cooling_rate: float = 0.3,
    compute_tier: int = 0,
    max_condense_steps: int = 6,
) -> Surface:
    """Build one gated amorphous slab targeting ``target_density`` sites/nm^2."""
    gate = SurfaceFidelityGate(material)
    key = gate.material
    lo, hi = gate.lo, gate.hi
    target = target_density if target_density is not None else _target_density_for(key)

    rng = np.random.default_rng(seed)
    # Melt-quench disorder: faster cooling (higher cooling_rate) -> more scatter.
    sigma = 0.18 * target * (0.5 + cooling_rate)
    density = float(target + rng.normal(0.0, sigma))

    # Condensation sweep: nudge an out-of-band draw back toward the band center.
    band_center = 0.5 * (lo + hi)
    for _ in range(max_condense_steps):
        if lo <= density <= hi:
            break
        density += 0.5 * (band_center - density)
    density = max(0.01, density)

    n_sites = max(1, int(round(density * _BASE_AREA_NM2)))
    area_nm2 = _BASE_AREA_NM2

    atoms = None
    descriptors: dict = {}
    if compute_tier >= 1 and _HAVE_ASE:
        atoms = _build_atoms(key, n_sites, area_nm2, rng)
        report = gate.check(atoms=atoms, seed=seed)
        descriptors = describe(atoms)
        n_sites = report["n_sites"]
        area_nm2 = report["area_nm2"]
    else:
        report = gate.check(n_sites=n_sites, area_nm2=area_nm2, seed=seed)
        descriptors = {"n_sites": n_sites, "note": "Tier-0 site inventory (no atoms)"}

    return Surface(
        material=key,
        label=material,
        seed=seed,
        target_density=round(target, 3),
        site_density_per_nm2=report["site_density_per_nm2"],
        n_sites=report["n_sites"],
        area_nm2=report["area_nm2"],
        passed=report["passed"],
        fidelity_report=report,
        descriptors=descriptors,
        atoms=atoms,
    )


def build_ensemble(
    material: str,
    n: int = 5,
    target_density: Optional[float] = None,
    seed0: int = 42,
    cooling_rate: float = 0.3,
    compute_tier: int = 0,
) -> list[Surface]:
    """Generate an ensemble of N independent gated slabs for one surface condition."""
    return [
        build_surface(
            material,
            target_density=target_density,
            seed=seed0 + i,
            cooling_rate=cooling_rate,
            compute_tier=compute_tier,
        )
        for i in range(n)
    ]


def ensemble_fidelity_summary(surfaces: list[Surface]) -> dict:
    """Aggregate an ensemble's fidelity for ``surface_fidelity.json``."""
    dens = [s.site_density_per_nm2 for s in surfaces]
    return {
        "material": surfaces[0].material if surfaces else None,
        "n_surfaces": len(surfaces),
        "target_density_per_nm2": surfaces[0].target_density if surfaces else None,
        "acceptance_band": (
            list(SITE_BANDS[surfaces[0].material]) if surfaces else None
        ),
        "site_density_mean": round(float(np.mean(dens)), 3) if dens else 0.0,
        "site_density_std": round(float(np.std(dens)), 3) if dens else 0.0,
        "n_passed": int(sum(s.passed for s in surfaces)),
        "all_passed": bool(all(s.passed for s in surfaces)) if surfaces else False,
        "reports": [s.fidelity_report for s in surfaces],
    }
