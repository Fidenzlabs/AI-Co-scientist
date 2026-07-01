"""Amorphous surface builder + fidelity gate (Deliverable #1, ADR-003)."""

from .amorphous_builder import (
    Surface,
    build_ensemble,
    build_surface,
    ensemble_fidelity_summary,
)
from .fidelity_gate import SITE_BANDS, SurfaceFidelityGate

__all__ = [
    "Surface",
    "build_surface",
    "build_ensemble",
    "ensemble_fidelity_summary",
    "SurfaceFidelityGate",
    "SITE_BANDS",
]

# Optional procedural-slab helpers (require pymatgen/ase); imported lazily by the builder.
try:  # pragma: no cover - convenience re-exports
    from .crystal_slabs import build_slab  # noqa: F401
    from .hydroxylation import saturate_surface  # noqa: F401

    __all__ += ["build_slab", "saturate_surface"]
except Exception:  # noqa: BLE001
    pass
