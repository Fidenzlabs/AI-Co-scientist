"""Random Sequential Adsorption (RSA) coverage (Phase 2, ADR-006).

Langmuir/blocking coverage assumes independent sites, but a bulky inhibitor sterically
excludes its neighbours, so the physically achievable packing is the RSA *jamming* limit,
not full monolayer. This caps the differential blocking coverage that drives selectivity
(exactly the mechanism behind the aniline ASD papers).

RSA of hard disks in 2D jams near ~0.547 area fraction; here we simulate it explicitly for
the inhibitor's footprint on the actual surface area, then cap the blocking coverage.
"""

from __future__ import annotations

import math

import numpy as np


def molecule_footprint_diameter_nm(atoms, vdw_pad_A: float = 1.6) -> float:
    """Estimate the lateral footprint diameter (nm) of an adsorbate from its geometry."""
    pos = atoms.get_positions()
    xy = pos[:, :2]
    center = xy.mean(axis=0)
    radial = np.linalg.norm(xy - center, axis=1)
    radius_A = float(radial.max()) + vdw_pad_A
    return max(0.2, 2.0 * radius_A / 10.0)


def rsa_saturation_coverage(
    box_nm: float,
    diam_nm: float,
    seed: int = 0,
    max_consec_rejections: int = 800,
) -> float:
    """Saturation (jamming) area fraction for RSA of hard disks in a periodic box."""
    if box_nm <= 0 or diam_nm <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    r = diam_nm / 2.0
    placed: list[np.ndarray] = []
    consec = 0
    d2 = diam_nm * diam_nm
    while consec < max_consec_rejections:
        p = rng.uniform(0.0, box_nm, size=2)
        ok = True
        for q in placed:
            dx = abs(p[0] - q[0]); dx = min(dx, box_nm - dx)
            dy = abs(p[1] - q[1]); dy = min(dy, box_nm - dy)
            if dx * dx + dy * dy < d2:
                ok = False
                break
        if ok:
            placed.append(p)
            consec = 0
        else:
            consec += 1
    covered = len(placed) * math.pi * r * r
    return float(min(1.0, covered / (box_nm * box_nm)))


def rsa_cap_fraction(atoms, area_nm2: float, seed: int = 0) -> float:
    """RSA jamming coverage fraction for ``atoms`` adsorbing on ``area_nm2`` of surface."""
    box_nm = math.sqrt(max(area_nm2, 1e-6))
    diam_nm = molecule_footprint_diameter_nm(atoms)
    return rsa_saturation_coverage(box_nm, diam_nm, seed=seed)


def apply_rsa_cap(blocking_coverage: float, rsa_fraction: float) -> float:
    """Cap ideal blocking coverage by the sterically achievable RSA jamming fraction."""
    return float(min(blocking_coverage, rsa_fraction))
