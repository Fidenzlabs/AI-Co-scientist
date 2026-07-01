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
