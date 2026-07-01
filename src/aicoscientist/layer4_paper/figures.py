"""Figure generation for the Layer-4 manuscript (ADR-007).

Renders the headline selectivity figure (S vs oxide thickness with the target line and
the ensemble band) directly from ``asald_results.json``. Degrades to returning ``None``
when matplotlib is unavailable, so the manuscript still compiles without the figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def selectivity_figure(rich: dict, out_path: Path) -> Path | None:
    """Plot S vs thickness from the validation results; return the image path or None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s); skipping figure", exc)
        return None

    sel = rich.get("selectivity", {})
    curve = sel.get("curve", {})
    thk_gs = curve.get("thk_gs_nm", [])
    s = curve.get("S", [])
    target = sel.get("target", 0.9)
    target_nm = sel.get("target_thickness_nm", 10.0)
    s_mean = sel.get("S_at_target_mean")
    s_std = sel.get("S_at_target_std", 0.0)
    if not thk_gs or not s:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thk_gs, s, "-", color="#1f77b4", label="S(thickness)")
    ax.axhline(target, ls="--", color="#d62728",
               label=f"target {target:.0%} @ {target_nm:g} nm")
    ax.axvline(target_nm, ls=":", color="gray")
    if s_mean is not None and s_std:
        ax.fill_between(
            thk_gs,
            [min(1.0, v + s_std) for v in s],
            [max(-1.0, v - s_std) for v in s],
            color="#1f77b4", alpha=0.15, label="ensemble band (±std)",
        )
    ax.set_xlabel("Growth-surface oxide thickness (nm)")
    ax.set_ylabel(r"Selectivity $S=(T_{GS}-T_{NGS})/(T_{GS}+T_{NGS})$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Area-selectivity vs film thickness")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
