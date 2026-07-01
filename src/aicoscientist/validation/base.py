"""Shared infrastructure for the specialist validator agents.

Each validator is an agent in the paper's Swarm: it takes a ``ValidationPlan`` and
returns a ``ValidationResult`` with quantitative metrics, a verdict, and a confidence.
This module provides the protocol plus shared verdict aggregation and RNG seeding so
runs are reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import (
    SuccessCriterion,
    ValidationMetric,
    ValidationPlan,
    ValidationResult,
    ValidationVerdict,
)


@runtime_checkable
class Validator(Protocol):
    """A specialist validation agent for one scientific domain."""

    domain: str

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        ...


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_criteria(
    metrics: list[ValidationMetric], criteria: list[SuccessCriterion]
) -> list[ValidationMetric]:
    """Annotate metrics with pass/fail against the plan's success criteria."""
    by_name = {m.name: m for m in metrics}
    for crit in criteria:
        metric = by_name.get(crit.metric)
        if metric is None:
            continue
        metric.threshold = crit.threshold
        metric.passed = crit.evaluate(metric.value)
    return metrics


def verdict_from_criteria(
    metrics: list[ValidationMetric], criteria: list[SuccessCriterion]
) -> tuple[ValidationVerdict, float]:
    """Aggregate criteria outcomes into a verdict and a confidence score."""
    checked = [m for m in metrics if m.passed is not None]
    if not checked:
        return ValidationVerdict.INCONCLUSIVE, 0.4

    passed = sum(1 for m in checked if m.passed)
    total = len(checked)
    frac = passed / total

    if frac >= 0.999:
        verdict = ValidationVerdict.SUPPORTED
    elif frac <= 0.001:
        verdict = ValidationVerdict.REJECTED
    else:
        verdict = ValidationVerdict.PARTIALLY_SUPPORTED

    # Confidence: how decisively the criteria were met/missed, blended with the
    # fraction passed so unanimous outcomes are more confident.
    decisiveness = abs(frac - 0.5) * 2
    confidence = round(min(0.97, 0.5 + 0.45 * decisiveness), 3)
    return verdict, confidence


def metric(name: str, value: float, unit: str = "", note: str = "") -> ValidationMetric:
    return ValidationMetric(name=name, value=float(value), unit=unit, note=note)
