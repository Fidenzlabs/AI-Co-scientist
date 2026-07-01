"""Surface fidelity gate (Deliverable #1, ADR-003).

Gate every generated slab on its EXPERIMENTAL surface-site density before it is allowed
into a reactivity calculation. This is the single highest-leverage rigor step: the brief
warns that modeled a-SiOx over-counts reactive sites and a-SiNx sites sit at irregular
spacing, so computed selectivity is dominated by the assumed surface. Rejecting
pathological slabs both raises credibility and *reduces* wasted compute.

Works on an ASE ``Atoms`` slab (real silanol / amine site counting) or on a supplied
site count + area, so it runs with or without ASE installed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Experimental acceptance bands (sites / nm^2).
#   SiO2: silanol OH/nm^2 after dehydroxylation; Zhuravlev ~1.15, controlled 0.35-2.0.
#   SiN:  -NH2/-NH surface amine density (broader; set per literature).
SITE_BANDS: dict[str, tuple[float, float]] = {
    "SiO2": (0.35, 2.5),
    "SiN": (1.0, 6.0),
}


def _material_key(material: str) -> str:
    """Map 'a-SiO2', 'SiOx', 'a-SiN', 'SiNx' onto a band key."""
    m = material.lower().replace("-", "").replace("a", "", 1) if material.lower().startswith("a-") else material.lower()
    if "sio" in material.lower() or "silica" in material.lower() or "oxide" in material.lower():
        return "SiO2"
    if "sin" in material.lower() or "nitride" in material.lower():
        return "SiN"
    if material in SITE_BANDS:
        return material
    raise ValueError(f"no acceptance band for material '{material}'")


class SurfaceFidelityGate:
    def __init__(self, material: str):
        self.key = _material_key(material)
        self.material = self.key
        self.lo, self.hi = SITE_BANDS[self.key]

    @staticmethod
    def _bonded(atoms, i, j, scale: float = 1.2) -> bool:
        from ase.data import covalent_radii

        d = atoms.get_distance(i, j, mic=True)
        rcut = scale * (covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]])
        return d < rcut

    def count_sites(self, atoms) -> tuple[int, float]:
        """Return (n_sites, area_nm2).

        Silanol = O-H whose O binds exactly one Si. Amine = N-H group. Simple neighbor
        heuristic; swap in ``ase.neighborlist`` for production accuracy.
        """
        from ase.data import atomic_numbers

        H = atomic_numbers["H"]
        O = atomic_numbers["O"]
        N = atomic_numbers["N"]
        Si = atomic_numbers["Si"]
        cell = atoms.get_cell()
        area_A2 = np.linalg.norm(np.cross(cell[0], cell[1]))
        area_nm2 = area_A2 / 100.0
        nums = atoms.numbers
        target_heavy = O if self.material == "SiO2" else N
        n_sites = 0
        heavy_idx = [k for k, z in enumerate(nums) if z == target_heavy]
        h_idx = [k for k, z in enumerate(nums) if z == H]
        for o in heavy_idx:
            has_h = any(self._bonded(atoms, o, h) for h in h_idx)
            if not has_h:
                continue
            if self.material == "SiO2":
                n_si = sum(self._bonded(atoms, o, s) for s in range(len(nums)) if nums[s] == Si)
                if n_si == 1:  # silanol, not bridging siloxane
                    n_sites += 1
            else:
                n_sites += 1  # any N-H counts as a reactive amine site
        return n_sites, area_nm2

    def check(
        self,
        atoms=None,
        n_sites: Optional[int] = None,
        area_nm2: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if atoms is not None:
            n_sites, area_nm2 = self.count_sites(atoms)
        if n_sites is None or area_nm2 is None:
            raise ValueError("provide an ASE Atoms slab, or n_sites and area_nm2")
        density = n_sites / area_nm2 if area_nm2 else 0.0
        passed = self.lo <= density <= self.hi
        return {
            "material": self.material,
            "site_density_per_nm2": round(density, 3),
            "acceptance_band": [self.lo, self.hi],
            "n_sites": int(n_sites),
            "area_nm2": round(area_nm2, 3),
            "passed": bool(passed),
            "seed": seed,
        }
