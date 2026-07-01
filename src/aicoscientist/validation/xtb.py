"""Tier-2 semi-empirical (GFN2-xTB) spot-checks (Phase 2, ADR-009).

A cheap, independent cross-check of the Tier-1 MLIP adsorption energies. We recompute a
small subset of adsorption configurations with GFN2-xTB (via ``tblite``) and feed the
resulting dE into the ``calibration_vs_literature`` block. xTB is not DFT, but a large
MLIP-vs-xTB disagreement is a useful red flag that the MLIP energies are untrustworthy for
that system.

Everything degrades gracefully: if ``tblite`` is not installed, :func:`xtb_available`
returns ``False`` and callers skip the spot-check.
"""

from __future__ import annotations


def xtb_available() -> bool:
    """True when the ``tblite`` ASE interface is importable."""
    try:
        import tblite.ase  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def make_xtb_calculator(method: str = "GFN2-xTB"):
    """Return a TBLite ASE calculator for the requested GFN method."""
    from tblite.ase import TBLite

    return TBLite(method=method)


def spotcheck_dE(
    slab,
    molecule,
    material_key: str,
    n_sites: int = 1,
    n_rot: int = 1,
    heights=(2.4,),
    method: str = "GFN2-xTB",
) -> dict:
    """Recompute the (subset) adsorption search with GFN2-xTB; return dE + configs."""
    from .mlip import adsorption_energy_search

    calc = make_xtb_calculator(method)
    res = adsorption_energy_search(
        slab, molecule, calc, material_key,
        n_sites=n_sites, n_rot=n_rot, heights=heights,
    )
    res["method"] = method
    return res
