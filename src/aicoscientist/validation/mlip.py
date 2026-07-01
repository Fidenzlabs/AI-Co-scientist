"""Foundation-MLIP hooks for the surface-reactivity engine (Tier 1, ADR-004).

Correct ASE / MACE APIs; meant to run on a GPU box (Colab CUDA) or CPU. Energies from
foundation MLIPs are only meaningful as DIFFERENCES within a *single* calculator + head
+ dispersion + dtype -- so do NOT mix ``mace_mp()`` and ``MACECalculator()``, and keep
those settings fixed across slab / molecule / complex. Adsorption energy is therefore
always ``E(slab+mol) - E(slab) - E(mol_gas)``, never a raw absolute.

All functions raise if the (optional) MLIP stack is unavailable; the ``surface_reactivity``
engine catches that and falls back to Tier-0 literature adsorption energies.

Ref: MACE (github.com/ACEsuit/mace); barrier caveat arXiv:2502.15582.
"""

from __future__ import annotations


def make_calculator(kind: str = "mace-mp", device: str = "cpu"):
    """Return an ASE calculator.

    * ``mace-mp``  -- MACE-MP medium, ungated/pip-installable (good default).
    * ``mace-mh1`` -- MACE-MH-1 (head='omat_pbe'); adds OC20 surface-adsorption and
      reaction-TS coverage, better for barriers.
    * ``chgnet``   -- CHGNet universal potential.
    """
    if kind == "mace-mp":
        from mace.calculators import mace_mp

        return mace_mp(
            model="medium", dispersion=True, default_dtype="float64", device=device
        )
    if kind == "mace-mh1":
        from mace.calculators import mace_mp

        return mace_mp(
            model="mace-mh-1", head="omat_pbe", default_dtype="float64", device=device
        )
    if kind == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(use_device=device, on_isolated_atoms="ignore")
    raise ValueError(f"unknown calculator {kind}")


def relax(atoms, calc, fmax: float = 0.05, steps: int = 300, fix_bottom_frac: float = 0.5):
    """Relax a slab/complex; freeze the bottom fraction to mimic bulk."""
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc
    z = atoms.positions[:, 2]
    cut = z.min() + fix_bottom_frac * (z.max() - z.min())
    atoms.set_constraint(FixAtoms(mask=[zi < cut for zi in z]))
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def adsorption_energy(slab, molecule, calc, place_height: float = 2.2, site_xy=None) -> dict:
    """dE_ads = E(slab+mol) - E(slab) - E(mol_gas). Negative => binding.

    Strong (< ~-0.7 eV) => chemisorption; weak (> ~-0.3 eV) => physisorption.
    """
    from ase.build import add_adsorbate
    from ase.optimize import LBFGS

    slab_r = relax(slab, calc)
    e_slab = slab_r.get_potential_energy()

    mol_r = molecule.copy()
    mol_r.calc = calc
    LBFGS(mol_r, logfile=None).run(fmax=0.03, steps=200)
    e_mol = mol_r.get_potential_energy()

    complex_ = slab_r.copy()
    if site_xy is None:
        site_xy = (slab_r.cell[0, 0] * 0.5, slab_r.cell[1, 1] * 0.5)
    add_adsorbate(complex_, molecule, height=place_height, position=site_xy)
    complex_r = relax(complex_, calc)
    e_complex = complex_r.get_potential_energy()

    dE = e_complex - e_slab - e_mol
    return {
        "dE_ads_eV": round(float(dE), 4),
        "regime": (
            "chemisorption" if dE < -0.7
            else ("physisorption" if dE > -0.3 else "intermediate")
        ),
    }


def barrier_neb(initial, final, calc, n_images: int = 7, fmax: float = 0.07) -> dict:
    """Optional Tier-2: precursor first-half-reaction barrier via NEB.

    NOTE: foundation MLIPs UNDERESTIMATE barriers (arXiv:2502.15582) -- treat the number
    as a lower bound and calibrate against literature DFT before reporting.
    """
    try:
        from ase.mep import NEB
    except ImportError:
        from ase.neb import NEB
    from ase.optimize import LBFGS

    images = [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]
    for im in images:
        im.calc = calc
    neb = NEB(images, climb=True)
    neb.interpolate()
    LBFGS(neb, logfile=None).run(fmax=fmax, steps=300)
    energies = [im.get_potential_energy() for im in images]
    ea = max(energies) - energies[0]
    return {
        "barrier_eV": round(float(ea), 4),
        "path_energies_eV": [round(e, 4) for e in energies],
    }


def build_molecule(name: str):
    """Best-effort small-molecule builder for a named inhibitor/precursor.

    Uses ASE's G2 database when the name matches; otherwise raises so the caller can
    fall back to Tier-0 literature adsorption energies.
    """
    from ase.build import molecule as ase_molecule

    aliases = {
        "acetic acid": "CH3COOH",
        "carboxylic acid": "HCOOH",
        "formic acid": "HCOOH",
        "methanesulfonic acid": "CH3SH",  # coarse stand-in in G2
        "aniline": "C6H6",                # aromatic stand-in
    }
    key = aliases.get(name.lower(), name)
    return ase_molecule(key)
