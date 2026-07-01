"""Supervisor routing table over the Swarm of specialist validator agents.

The AS-ALD co-scientist ships a single materials-reactivity engine; the registry keeps
the pluggable shape (keyword -> engine) so additional engines can be added later without
touching the Layer-3 graph.
"""

from __future__ import annotations

from .base import Validator
from .surface_reactivity import SurfaceReactivityValidator

# Keywords that route a hypothesis to the surface-reactivity engine (ADR-004).
SURFACE_REACTIVITY_CUES = (
    "passivate", "selective deposition", "area-selective", "ald", "inhibitor",
    "nitride", "oxide", "precursor", "chemisorb", "physisorb", "selectivity",
    "surface", "silica", "silanol",
)


def build_registry() -> dict[str, Validator]:
    return {"surface_reactivity": SurfaceReactivityValidator()}


def get_validator(domain: str) -> Validator:
    registry = build_registry()
    return registry.get(domain, registry["surface_reactivity"])
