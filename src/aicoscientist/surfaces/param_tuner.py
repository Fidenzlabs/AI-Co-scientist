"""LLM param-tuning agent for the MLIP melt-quench surface builder.

Closes a loop around the melt-quench MD: build a small, cheap probe slab with the current
(melt_T, quench_steps), score it against the fidelity gate + Si coordination quality, hand
the numbers to an LLM that proposes the next (melt_T, quench_steps) or stops, and keep the
best. The tuned params are then reused for the full ensemble build.

Grounded in the melt-quench-MD literature: a-SiO2 melt ~4000 K, a-Si3N4 ~2500-5000 K; NVT
quench (uMLIPs over-expand under NPT); a-Si3N4 RDF is largely quench-rate insensitive over
1e13-1e15 K/s. The agent nudges within physically sane bounds; a deterministic heuristic is
used when no LLM key is configured (offline).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Physically sane bounds the agent (and heuristic) must stay within.
_MELT_T_LO, _MELT_T_HI = 2000.0, 5000.0
_QUENCH_LO, _QUENCH_HI = 500, 40000


class MeltQuenchProposal(BaseModel):
    """One LLM step: the next melt-quench params to try, or a stop signal."""

    melt_temperature_k: float = Field(description="next melt temperature in Kelvin")
    quench_steps: int = Field(description="next number of quench MD steps")
    stop: bool = Field(description="true if the current best is good enough to stop")
    rationale: str = Field(description="one-line reason for this move")


def _coordination_metrics(atoms, key: str) -> dict:
    """Si coordination quality: mean CN, tetrahedral fraction, defect fraction.

    A well-formed a-SiO2 / a-Si3N4 network is ~4-coordinated Si; deviations flag dangling
    bonds or over-coordination (a common uMLIP high-T pathology)."""
    import numpy as np
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoffs = natural_cutoffs(atoms, mult=1.15)
    i = neighbor_list("i", atoms, cutoffs)
    counts = np.bincount(i, minlength=len(atoms))
    numbers = np.asarray(atoms.get_atomic_numbers())
    si = numbers == 14
    if not si.any():
        return {"si_mean_cn": 0.0, "si_tetrahedral_frac": 0.0, "si_defect_frac": 1.0}
    si_cn = counts[si]
    return {
        "si_mean_cn": round(float(np.mean(si_cn)), 3),
        "si_tetrahedral_frac": round(float(np.mean(si_cn == 4)), 3),
        "si_defect_frac": round(float(np.mean(si_cn != 4)), 3),
    }


def _score(metrics: dict, gate_passed: bool) -> float:
    """Higher is better: reward tetrahedral Si + gate pass, penalize coordination defects."""
    if "error" in metrics:
        return -1e9
    return (
        metrics.get("si_tetrahedral_frac", 0.0)
        - 0.5 * metrics.get("si_defect_frac", 1.0)
        + (0.15 if gate_passed else 0.0)
    )


def _clamp(melt_t: float, quench: int) -> tuple[float, int]:
    return (
        float(min(max(melt_t, _MELT_T_LO), _MELT_T_HI)),
        int(min(max(quench, _QUENCH_LO), _QUENCH_HI)),
    )


def _heuristic_next(key: str, history: list[dict]) -> MeltQuenchProposal:
    """Deterministic fallback when no LLM is available.

    If Si is over-coordinated (mean CN > 4) the melt is too hot / quench too fast -> cool
    and quench longer; if under-coordinated (< 4, dangling) -> hotter melt to re-network.
    Stop when the gate passes with a low defect fraction."""
    last = history[-1]
    m = last["metrics"]
    melt_t, quench = last["melt_T"], last["quench"]
    if last["gate_passed"] and m.get("si_defect_frac", 1.0) <= 0.15:
        return MeltQuenchProposal(melt_temperature_k=melt_t, quench_steps=quench,
                                  stop=True, rationale="gate passed, low defects")
    cn = m.get("si_mean_cn", 4.0)
    if cn > 4.2:
        melt_t, quench = _clamp(melt_t - 400, int(quench * 1.5))
        why = f"Si over-coordinated (CN={cn}) -> cooler melt, longer quench"
    else:
        melt_t, quench = _clamp(melt_t + 300, quench)
        why = f"Si under-coordinated (CN={cn}) -> hotter melt to re-network"
    return MeltQuenchProposal(melt_temperature_k=melt_t, quench_steps=quench,
                              stop=False, rationale=why)


def _propose_next(key: str, history: list[dict], trials: int) -> MeltQuenchProposal:
    """Ask the LLM for the next move; fall back to the heuristic on any failure."""
    try:
        from ..llm import structured_call

        system = (
            "You tune melt-quench molecular-dynamics parameters for generating an amorphous "
            f"{key} surface with a foundation MLIP (MACE), NVT ensemble. Goal: a well-formed "
            "~4-coordinated Si network (high tetrahedral fraction, low defect fraction) that "
            "passes the site-density fidelity gate. Literature: a-SiO2 melt ~4000 K, a-Si3N4 "
            "~2500-5000 K; over-coordination/expansion means too hot or too fast a quench; "
            "dangling bonds mean under-melted. Stay within melt_T [2000,5000] K and "
            "quench_steps [500,40000]. Propose the next params or stop if the best is good."
        )
        user = (
            f"Material: {key}. Trial budget: {trials}. History (oldest->newest):\n"
            + "\n".join(
                f"- trial {h['trial']}: melt_T={h['melt_T']}K quench={h['quench']} "
                f"metrics={h['metrics']} gate_passed={h['gate_passed']} score={h['score']}"
                for h in history
            )
            + "\nReturn the next melt_temperature_k, quench_steps, stop, rationale."
        )
        prop = structured_call(MeltQuenchProposal, system, user)
        melt_t, quench = _clamp(prop.melt_temperature_k, prop.quench_steps)
        prop.melt_temperature_k, prop.quench_steps = melt_t, quench
        return prop
    except Exception as exc:  # noqa: BLE001 -- offline / no key / parse error
        logger.info("melt-quench tuner: LLM unavailable (%s); using heuristic", exc)
        return _heuristic_next(key, history)


def tune_melt_quench(key: str, seed: int, miller, supercell, settings) -> dict:
    """Iterate melt_T / quench_steps on a cheap probe slab; return the best-scoring params.

    Returns ``{"melt_temperature_k", "quench_steps", "history"}``. Never raises: a failed
    probe scores -inf and the loop continues; if everything fails the config defaults win.
    """
    from .amorphous_builder import _md_amorphous_slab, _target_density_for
    from .descriptors import describe
    from .fidelity_gate import SurfaceFidelityGate

    gate = SurfaceFidelityGate(key)
    key = gate.material
    try:
        probe_sc = tuple(int(x) for x in str(settings.mq_autotune_probe_supercell).split(","))
    except Exception:  # noqa: BLE001
        probe_sc = (2, 2)
    probe_quench_cap = int(settings.mq_autotune_probe_quench)
    trials = max(1, int(settings.mq_autotune_trials))
    target = _target_density_for(key)

    melt_t = float(settings.mq_melt_temperature_k)
    quench = int(settings.mq_quench_steps)
    history: list[dict] = []
    best = {"melt_temperature_k": melt_t, "quench_steps": quench, "score": -1e18}

    for t in range(trials):
        overrides = {"melt_temperature_k": melt_t,
                     "quench_steps": min(quench, probe_quench_cap)}
        try:
            atoms, _prov = _md_amorphous_slab(key, target, seed, miller, probe_sc,
                                              settings, overrides=overrides)
            metrics = _coordination_metrics(atoms, key)
            rep = gate.check(atoms=atoms, seed=seed, descriptors=describe(atoms))
            gate_passed = bool(rep.get("passed", False))
        except Exception as exc:  # noqa: BLE001
            metrics, gate_passed = {"error": str(exc)}, False
        sc = _score(metrics, gate_passed)
        history.append({"trial": t, "melt_T": melt_t, "quench": quench,
                        "metrics": metrics, "gate_passed": gate_passed, "score": round(sc, 3)})
        logger.info("melt-quench tune %s trial %d: melt_T=%.0f quench=%d score=%.3f %s",
                    key, t, melt_t, quench, sc, metrics)
        if sc > best["score"]:
            best = {"melt_temperature_k": melt_t, "quench_steps": quench, "score": sc}
        if t == trials - 1:
            break
        prop = _propose_next(key, history, trials)
        if prop.stop:
            break
        melt_t, quench = float(prop.melt_temperature_k), int(prop.quench_steps)

    return {"melt_temperature_k": best["melt_temperature_k"],
            "quench_steps": best["quench_steps"], "history": history}


@lru_cache(maxsize=8)
def tuned_overrides(key: str, miller, supercell, sig) -> dict:
    """Cached per (material, geometry, settings-signature) so we tune once, not per slab.

    ``sig`` is a hashable tuple of the settings that affect tuning, so changing them
    triggers a retune. ``settings`` is fetched fresh here to keep the cache key simple.
    """
    from ..config import get_settings

    settings = get_settings()
    # Tune on a fixed probe seed for reproducibility.
    result = tune_melt_quench(key, seed=12345, miller=miller, supercell=supercell,
                              settings=settings)
    return {"melt_temperature_k": result["melt_temperature_k"],
            "quench_steps": result["quench_steps"]}
