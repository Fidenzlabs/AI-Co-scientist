"""Procedural crystalline-derived slab builder (Phase 1, ADR-003).

Builds reproducible slabs from crystalline bulk phases with pymatgen's ``SlabGenerator``
and hands them back as ASE ``Atoms``:

* ``SiO2`` -- alpha-quartz (space group 152, P3_1 2 1).
* ``SiN``  -- beta-Si3N4 (space group 176, P6_3/m).

These are *crystalline-derived* hydroxylatable surfaces, not fully amorphous networks
(true melt-quench amorphization is a later Phase-3 step). They give a real bulk + real
under-coordinated surface Si to cap with -OH / -NH2, which is what makes MLIP adsorption
energies meaningful. Everything is generated from lattice + Wyckoff data, so no external
CIF downloads are required and runs are reproducible.
"""

from __future__ import annotations

import numpy as np

# Bulk crystal specs using ASE's documented spacegroup recipes (known-good stoichiometry).
#   SiO2 -> alpha-quartz, SG 154 (P3_2 2 1); yields Si3O6 per cell.
#   SiN  -> beta-Si3N4,   SG 176 (P6_3/m);   yields Si6N8 per cell.
_THIRD = 1.0 / 3.0
_TWOTHIRD = 2.0 / 3.0
_BULK_SPECS: dict[str, dict] = {
    "SiO2": {
        "spacegroup": 152,
        "a": 4.9137,
        "c": 5.4047,
        "species": ["Si", "O"],
        # Si 3a + O 6c giving proper SiO4 tetrahedra (Si-O ~1.56-1.64 A).
        "coords": [[0.4697, 0.0, _THIRD], [0.4135, 0.2669, 0.20]],
        "phase": "alpha-quartz",
    },
    "SiN": {
        "spacegroup": 176,
        "a": 7.595,
        "c": 2.902,
        "species": ["Si", "N", "N"],
        "coords": [
            [0.1736, 0.7687, 0.25],
            [0.3300, 0.0344, 0.25],
            [_THIRD, _TWOTHIRD, 0.25],
        ],
        "phase": "beta-Si3N4",
    },
}


def build_bulk_atoms(material_key: str):
    """Return an ASE ``Atoms`` bulk crystal via ASE's spacegroup builder."""
    from ase.spacegroup import crystal

    spec = _BULK_SPECS[material_key]
    return crystal(
        spec["species"],
        basis=spec["coords"],
        spacegroup=spec["spacegroup"],
        cellpar=[spec["a"], spec["a"], spec["c"], 90, 90, 120],
    )


def build_bulk(material_key: str):
    """Return a pymatgen ``Structure`` for the crystalline bulk phase."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(build_bulk_atoms(material_key))


def build_slab(
    material_key: str,
    miller_index: tuple[int, int, int] = (1, 0, 0),
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 14.0,
    supercell: tuple[int, int] = (2, 2),
):
    """Generate an ASE slab (pbc in a/b, vacuum in c) for the requested surface.

    Returns ``(atoms, provenance)`` where provenance records the phase, Miller index and
    slab dimensions for the manuscript methods table.
    """
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    bulk = build_bulk(material_key)
    gen = SlabGenerator(
        bulk,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=True,
        primitive=True,
        lll_reduce=True,
    )
    slab = gen.get_slab()
    if supercell and supercell != (1, 1):
        slab.make_supercell([[supercell[0], 0, 0], [0, supercell[1], 0], [0, 0, 1]])

    atoms = AseAtomsAdaptor.get_atoms(slab)
    atoms.set_pbc([True, True, False])

    cell = atoms.get_cell()
    area_nm2 = float(np.linalg.norm(np.cross(cell[0], cell[1]))) / 100.0
    provenance = {
        "source": "procedural-crystalline",
        "phase": _BULK_SPECS[material_key]["phase"],
        "miller_index": list(miller_index),
        "supercell": list(supercell),
        "n_atoms": len(atoms),
        "area_nm2": round(area_nm2, 3),
    }
    return atoms, provenance
