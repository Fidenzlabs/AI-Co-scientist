"""Protein structure validator — delegates to structure_based_design.

Kept for backward compatibility with plans that route to the ``protein`` domain.
Real structure-based drug design is handled by the DiffSBDD-inspired validator.
"""

from __future__ import annotations

from pathlib import Path

from ..models import ValidationPlan, ValidationResult
from .structure_based_design import StructureBasedDesignValidator

_delegate = StructureBasedDesignValidator()


class ProteinValidator:
    domain = "protein"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        return _delegate.run(run_id, hypothesis, plan, datasets_dir, logs_dir)
