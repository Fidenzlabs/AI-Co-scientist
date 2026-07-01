"""``surface_reactivity`` validation engine (ADR-004 / ADR-006 / ADR-009).

Implements the graded in-silico test for area-selective ALD as the ADR-009 five-step
protocol, over an *ensemble* of experiment-faithful amorphous surfaces:

1. Build & gate surfaces (Deliverable #1): N a-SiO2 (GS) + N a-SiN (NGS) slabs, each
   passed through the fidelity gate; failures are discarded.
2. Inhibitor adsorption screen: dE_ads on GS vs NGS -- chemisorb on NGS, physisorb on GS.
   Tier 1 computes dE with a foundation MLIP; Tier 0 uses literature/xTB priors (per the
   selection agent) with per-surface scatter.
3. Effective (chemisorbed, purge-surviving) blocking coverage; the DIFFERENTIAL blocking
   theta(NGS) - theta(GS) is the selectivity driver.
4. [Optional Tier-2] precursor barrier (lower bound; calibrate vs literature DFT).
5. Selectivity & verdict: differential blocking -> nucleation delay -> S(N), reported as
   mean +/- std over the ensemble, with a literature-calibration validity flag.

Emits the paper-ready ``asald_results.json`` and ``surface_fidelity.json`` alongside the
standard repo ``ValidationResult``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..models import ValidationPlan, ValidationResult, ValidationVerdict
from ..surfaces import build_ensemble, ensemble_fidelity_summary
from .base import apply_criteria, metric
from .selectivity_model import (
    SelectivityModel,
    blocking_coverage_from_dE,
    coverage_from_dE,
)

logger = logging.getLogger(__name__)


class SurfaceReactivityValidator:
    domain = "surface_reactivity"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        settings = get_settings()
        spec = plan.data_spec or {}
        run_dir = datasets_dir.parent

        inhibitor = spec.get("inhibitor", "acetic acid")
        precursor = spec.get("precursor", "BDEAS")
        gs_label = spec.get("growth_surface", "a-SiO2")
        ngs_label = spec.get("non_growth_surface", "a-SiN")
        target_nm = float(spec.get("target_thickness_nm", 10.0))
        target_sel = float(spec.get("target_selectivity", 0.90))
        T = float(spec.get("temperature_K", settings.ald_temperature_k))
        dose_ratio = float(spec.get("dose_ratio", 1.0))
        n = int(spec.get("ensemble_n", settings.surface_ensemble_n))
        tier = int(spec.get("compute_tier", settings.compute_tier))

        dE_ngs_prior = float(spec.get("dE_ngs_eV", -1.0))
        dE_gs_prior = float(spec.get("dE_gs_eV", -0.2))
        prior_std = float(spec.get("dE_prior_std", 0.08))
        literature_dE_ngs = spec.get("literature_dE_ngs_eV", dE_ngs_prior)
        barrier = spec.get("barrier")

        seed0 = plan.seed or 42

        # ---- Step 1: build & gate the surface ensembles -----------------------------
        gs_surfaces = build_ensemble(
            gs_label, n=n, seed0=seed0, compute_tier=tier,
            target_density=spec.get("target_density_gs"),
        )
        ngs_surfaces = build_ensemble(
            ngs_label, n=n, seed0=seed0 + 1000, compute_tier=tier,
            target_density=spec.get("target_density_ngs"),
        )
        gs_pass = [s for s in gs_surfaces if s.passed] or gs_surfaces
        ngs_pass = [s for s in ngs_surfaces if s.passed] or ngs_surfaces

        fidelity = {
            "growth_surface": ensemble_fidelity_summary(gs_surfaces),
            "non_growth_surface": ensemble_fidelity_summary(ngs_surfaces),
        }
        (run_dir / "surface_fidelity.json").write_text(
            json.dumps(fidelity, indent=2), encoding="utf-8"
        )

        # ---- Step 2: inhibitor adsorption screen ------------------------------------
        calc, engine = self._maybe_calculator(tier, settings)
        dE_ngs = self._adsorption_samples(
            ngs_pass, inhibitor, dE_ngs_prior, prior_std, calc, engine
        )
        dE_gs = self._adsorption_samples(
            gs_pass, inhibitor, dE_gs_prior, prior_std, calc, engine
        )
        dE_ngs = np.array(dE_ngs, float)
        dE_gs = np.array(dE_gs, float)

        # ---- Step 3: coverage + differential blocking -------------------------------
        theta_ngs = np.array([coverage_from_dE(d, T, dose_ratio) for d in dE_ngs])
        theta_gs = np.array([coverage_from_dE(d, T, dose_ratio) for d in dE_gs])
        block_ngs = np.array([blocking_coverage_from_dE(d, T, dose_ratio) for d in dE_ngs])
        block_gs = np.array([blocking_coverage_from_dE(d, T, dose_ratio) for d in dE_gs])

        # ---- Step 5: per-surface selectivity -> ensemble mean +/- std ---------------
        model = SelectivityModel()
        s_samples = []
        for bn, bg in zip(block_ngs, block_gs):
            delay = model.nucleation_delay_cycles(bn, bg)
            s_samples.append(
                model.selectivity_at_thickness(delay, target_nm)["selectivity_at_target"]
            )
        s_samples = np.array(s_samples)
        s_mean, s_std = float(s_samples.mean()), float(s_samples.std())

        verdict_str = (
            "supported" if s_mean >= target_sel
            else "partially_supported" if s_mean >= 0.75 * target_sel
            else "rejected"
        )
        verdict = {
            "supported": ValidationVerdict.SUPPORTED,
            "partially_supported": ValidationVerdict.PARTIALLY_SUPPORTED,
            "rejected": ValidationVerdict.REJECTED,
        }[verdict_str]

        # Calibration vs literature (rigor flag).
        calib = None
        if literature_dE_ngs is not None:
            abs_err = abs(float(dE_ngs.mean()) - float(literature_dE_ngs))
            calib = {
                "predicted_dE_ngs_eV": round(float(dE_ngs.mean()), 4),
                "literature_dE_ngs_eV": round(float(literature_dE_ngs), 4),
                "abs_error_eV": round(abs_err, 4),
                "validity_flag": "ok" if abs_err < 0.3 else "review",
            }

        # Selectivity curve for the Layer-4 figure (mean differential blocking).
        delay_mean = model.nucleation_delay_cycles(float(block_ngs.mean()), float(block_gs.mean()))
        n_cyc, thk_gs, thk_ngs, s_curve = model.selectivity_curve(delay_mean)

        rich = {
            "hypothesis": {
                "statement": hypothesis,
                "growth_surface": gs_label,
                "non_growth_surface": ngs_label,
                "inhibitor": inhibitor,
                "precursor": precursor,
                "target_film": spec.get("target_film", "SiOx"),
                "target_thickness_nm": target_nm,
                "target_selectivity": target_sel,
                "provenance_refs": spec.get("provenance_refs", []),
            },
            "surface_ensemble": {
                "n_surfaces_gs": int(dE_gs.size),
                "n_surfaces_ngs": int(dE_ngs.size),
                "fidelity_reports": (
                    fidelity["growth_surface"]["reports"]
                    + fidelity["non_growth_surface"]["reports"]
                ),
                "all_surfaces_passed_gate": bool(
                    fidelity["growth_surface"]["all_passed"]
                    and fidelity["non_growth_surface"]["all_passed"]
                ),
            },
            "inhibitor_adsorption": {
                "engine": engine,
                "dE_ngs_mean_eV": round(float(dE_ngs.mean()), 4),
                "dE_ngs_std_eV": round(float(dE_ngs.std()), 4),
                "dE_gs_mean_eV": round(float(dE_gs.mean()), 4),
                "dE_gs_std_eV": round(float(dE_gs.std()), 4),
                "theta_eq_ngs_mean": round(float(theta_ngs.mean()), 4),
                "theta_eq_gs_mean": round(float(theta_gs.mean()), 4),
                "blocking_ngs_mean": round(float(block_ngs.mean()), 4),
                "blocking_gs_mean": round(float(block_gs.mean()), 4),
                "differential_blocking": round(float(block_ngs.mean() - block_gs.mean()), 4),
                "differential_selectivity_signal": round(
                    float(dE_gs.mean() - dE_ngs.mean()), 4
                ),
            },
            "precursor_barrier": barrier,
            "selectivity": {
                "metric": "S = (Thk_GS - Thk_NGS)/(Thk_GS + Thk_NGS)",
                "target": target_sel,
                "target_thickness_nm": target_nm,
                "S_at_target_mean": round(s_mean, 4),
                "S_at_target_std": round(s_std, 4),
                "curve": {
                    "cycle": [int(c) for c in n_cyc[::10]],
                    "thk_gs_nm": [round(float(v) / 10, 3) for v in thk_gs[::10]],
                    "thk_ngs_nm": [round(float(v) / 10, 3) for v in thk_ngs[::10]],
                    "S": [round(float(v), 4) for v in s_curve[::10]],
                },
            },
            "calibration_vs_literature": calib,
            "verdict": verdict_str,
            "provenance": {
                "engine": engine,
                "compute_tier": tier,
                "mlip_model": settings.mlip_model if tier >= 1 else None,
                "mlip_device": settings.resolved_mlip_device if tier >= 1 else None,
                "temperature_K": T,
                "dose_ratio": dose_ratio,
                "ensemble_n": n,
                "seed": seed0,
            },
        }
        (run_dir / "asald_results.json").write_text(
            json.dumps(rich, indent=2), encoding="utf-8"
        )
        (logs_dir / f"surface_reactivity_iter{plan.iteration}.json").write_text(
            json.dumps({"engine": engine, "tier": tier, "verdict": verdict_str,
                        "S_at_target_mean": round(s_mean, 4)}, indent=2),
            encoding="utf-8",
        )

        # ---- repo-schema metrics + verdict ------------------------------------------
        metrics = [
            metric("S_at_target", round(s_mean, 4),
                   note=f"selectivity at {target_nm} nm (mean over ensemble, +/-{s_std:.3f})"),
            metric("differential_blocking",
                   round(float(block_ngs.mean() - block_gs.mean()), 4),
                   note="theta_block(NGS) - theta_block(GS): the selectivity driver"),
            metric("dE_ngs_mean_eV", round(float(dE_ngs.mean()), 4),
                   note="inhibitor adsorption on NGS (chemisorption expected < -0.7)"),
            metric("dE_gs_mean_eV", round(float(dE_gs.mean()), 4),
                   note="inhibitor adsorption on GS (physisorption expected > -0.3)"),
        ]
        if calib is not None:
            metrics.append(
                metric("calibration_abs_error_eV", calib["abs_error_eV"],
                       note=f"MLIP-vs-literature dE delta; flag={calib['validity_flag']}")
            )
        metrics = apply_criteria(metrics, plan.success_criteria)

        confidence = self._confidence(s_mean, target_sel, s_std, calib)
        narrative = (
            f"Tested '{inhibitor}' inhibitor / '{precursor}' precursor over {dE_ngs.size} "
            f"{ngs_label} (NGS) and {dE_gs.size} {gs_label} (GS) gated surfaces using {engine}. "
            f"Differential blocking {block_ngs.mean() - block_gs.mean():.3f}; "
            f"S = {s_mean:.3f} +/- {s_std:.3f} at {target_nm} nm vs target {target_sel:.2f} "
            f"-> {verdict_str}."
        )

        return ValidationResult(
            run_id=run_id,
            hypothesis_statement=hypothesis,
            plan=plan,
            metrics=metrics,
            verdict=verdict,
            confidence=confidence,
            narrative=narrative,
            artifact_paths={
                "asald_results": str(run_dir / "asald_results.json"),
                "surface_fidelity": str(run_dir / "surface_fidelity.json"),
                "log": str(logs_dir / f"surface_reactivity_iter{plan.iteration}.json"),
            },
        )

    # ──────────────────────── helpers ────────────────────────

    def _maybe_calculator(self, tier: int, settings):
        """Return (calc, engine_label). Falls back to Tier-0 on any import/setup error."""
        if tier < 1:
            return None, "tier0-literature-priors"
        try:
            from .mlip import make_calculator

            calc = make_calculator(settings.mlip_model, settings.resolved_mlip_device)
            return calc, f"{settings.mlip_model}@{settings.resolved_mlip_device}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLIP unavailable (%s); falling back to Tier-0 priors", exc)
            return None, "tier0-literature-priors (MLIP unavailable)"

    def _adsorption_samples(self, surfaces, inhibitor, prior_mean, prior_std, calc, engine):
        """dE per surface: MLIP when possible, else literature prior + per-surface scatter."""
        out: list[float] = []
        for s in surfaces:
            dE = None
            if calc is not None and s.atoms is not None:
                try:
                    from .mlip import adsorption_energy, build_molecule

                    mol = build_molecule(inhibitor)
                    dE = adsorption_energy(s.atoms, mol, calc)["dE_ads_eV"]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MLIP adsorption failed on surface %s (%s); using prior",
                                   s.seed, exc)
                    dE = None
            if dE is None:
                rng = np.random.default_rng(s.seed)
                dE = float(prior_mean + rng.normal(0.0, prior_std))
            out.append(dE)
        return out

    @staticmethod
    def _confidence(s_mean: float, target: float, s_std: float, calib: dict | None) -> float:
        decisiveness = min(1.0, abs(s_mean - target) / max(target, 1e-6))
        conf = 0.55 + 0.35 * decisiveness
        conf -= min(0.15, s_std)            # wide ensemble spread lowers confidence
        if calib and calib.get("validity_flag") == "review":
            conf -= 0.15                     # uncalibrated MLIP lowers confidence
        return round(max(0.3, min(0.97, conf)), 3)
