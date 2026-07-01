"""Selectivity + nucleation-delay scoring (ADR-006).

Maps microscopic reactivity (inhibitor adsorption energies) onto the brief's metric
``S = (Thk_GS - Thk_NGS) / (Thk_GS + Thk_NGS)`` via a reduced-order nucleation-delay
model. Ported from the verified reference implementation; the coverage physics is the
subtle part: only *chemisorbed, purge-surviving* inhibitor blocks the precursor, and the
selectivity driver is the DIFFERENTIAL blocking coverage, not raw Langmuir coverage
(which saturates at ALD temperature and would wash out selectivity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K


def coverage_from_dE(
    dE_ads_eV: float, T: float = 423.0, partial_pressure_ratio: float = 1.0
) -> float:
    """Equilibrium Langmuir coverage.

    NOTE: at ALD temperatures this saturates even for weak binding, so it is NOT the
    selectivity driver -- use :func:`blocking_coverage_from_dE`.
    """
    K = math.exp(-dE_ads_eV / (KB_EV * T))
    Kp = K * partial_pressure_ratio
    return Kp / (1.0 + Kp)


def blocking_coverage_from_dE(
    dE_ads_eV: float,
    T: float = 423.0,
    partial_pressure_ratio: float = 1.0,
    E_chem: float = 0.5,
    width: float = 0.1,
) -> float:
    """EFFECTIVE blocking coverage.

    Only CHEMISORBED inhibitor survives the ALD purge and blocks the precursor;
    physisorbed molecules desorb and confer no selectivity. We gate the equilibrium
    coverage by a chemisorption-survival sigmoid on ``|dE|`` (physisorption |dE| < E_chem
    -> ~0 blocking; chemisorption -> ~1). This is the physically correct driver of area
    selectivity (aniline: chemisorb NGS, physisorb GS).
    """
    theta_eq = coverage_from_dE(dE_ads_eV, T, partial_pressure_ratio)
    survival = 1.0 / (1.0 + math.exp(-(-dE_ads_eV - E_chem) / width))
    return theta_eq * survival


@dataclass
class SelectivityModel:
    """Reduced-order nucleation-delay model.

    Calibrate ``delay_gain`` against a known ASD system (e.g. aniline: ~6 nm selective
    growth) before trusting absolute cycle counts.
    """

    gpc_gs_A: float = 1.0             # growth-per-cycle on clean GS (Angstrom)
    gpc_ngs_A: float = 1.0            # intrinsic GPC on NGS after breakthrough
    gpc_ngs_residual_A: float = 0.04  # background defect nucleation on blocked NGS (never 0)
    delay_gain: float = 115.0         # cycles of delay at full differential blocking

    def nucleation_delay_cycles(self, block_ngs: float, block_gs: float) -> float:
        """Delay scales with the DIFFERENTIAL blocking coverage (NGS blocked, GS open)."""
        return self.delay_gain * max(0.0, block_ngs - block_gs)

    def thickness(self, n_cycles: np.ndarray, delay: float):
        thk_gs = self.gpc_gs_A * n_cycles
        within = np.clip(n_cycles, 0, delay)             # blocked window: residual growth
        beyond = np.clip(n_cycles - delay, 0, None)      # after breakthrough: full growth
        thk_ngs = self.gpc_ngs_residual_A * within + self.gpc_ngs_A * beyond
        return thk_gs, thk_ngs

    def selectivity_curve(self, delay: float, max_cycles: int = 400):
        n = np.arange(0, max_cycles + 1, dtype=float)
        thk_gs, thk_ngs = self.thickness(n, delay)
        denom = np.where((thk_gs + thk_ngs) > 0, thk_gs + thk_ngs, 1.0)
        S = (thk_gs - thk_ngs) / denom
        return n, thk_gs, thk_ngs, S

    def selectivity_at_thickness(self, delay: float, target_nm: float) -> dict:
        target_A = target_nm * 10.0
        n_star = target_A / self.gpc_gs_A
        thk_gs, thk_ngs = self.thickness(np.array([n_star]), delay)
        s = float((thk_gs[0] - thk_ngs[0]) / max(thk_gs[0] + thk_ngs[0], 1e-9))
        return {
            "cycles_to_target": round(float(n_star), 1),
            "thk_gs_nm": round(float(thk_gs[0]) / 10, 3),
            "thk_ngs_nm": round(float(thk_ngs[0]) / 10, 3),
            "selectivity_at_target": round(s, 4),
        }
