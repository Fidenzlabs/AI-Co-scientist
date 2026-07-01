"""Supervisor routing table over the Swarm of specialist validator agents."""

from __future__ import annotations

from .base import Validator
from .cheminformatics import CheminformaticsValidator
from .drug_repurposing import DrugRepurposingValidator
from .mechanistic import MechanisticValidator
from .protein import ProteinValidator
from .statistical import StatisticalValidator
from .structure_based_design import StructureBasedDesignValidator


def build_registry() -> dict[str, Validator]:
    sbdd = StructureBasedDesignValidator()
    return {
        "statistical": StatisticalValidator(),
        "cheminformatics": CheminformaticsValidator(),
        "mechanistic": MechanisticValidator(),
        "drug_repurposing": DrugRepurposingValidator(),
        "structure_based_design": sbdd,
        "protein": sbdd,  # alias -> DiffSBDD-inspired validator
    }


def get_validator(domain: str) -> Validator:
    registry = build_registry()
    return registry.get(domain, registry["statistical"])
