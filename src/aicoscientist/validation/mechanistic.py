"""Mechanistic / systems-biology validator (REAL ODE simulation via scipy).

Integrates a small dynamical model (two-compartment PK, logistic growth, or a negative
feedback loop / digital-twin style system) and derives quantitative behavior: AUC, peak,
steady state, and parameter sensitivity. These feed the plan's success criteria.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from ..models import ValidationPlan, ValidationResult
from .base import apply_criteria, metric, verdict_from_criteria


class MechanisticValidator:
    domain = "mechanistic"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        spec = plan.data_spec or {}
        model = spec.get("model", "two_compartment")
        params = dict(spec.get("params", {}))
        t_end = float(spec.get("t_end", 24.0))
        n_points = int(spec.get("n_points", 200))
        t_eval = np.linspace(0.0, t_end, n_points)

        t, y, observable = self._simulate(model, params, t_end, t_eval)

        auc = float(np.trapz(observable, t))
        peak = float(np.max(observable))
        final = float(observable[-1])
        # Steady state: last 10% of the trace is ~flat.
        tail = observable[max(1, int(0.9 * len(observable))):]
        rel_drift = float(np.std(tail) / (np.mean(np.abs(tail)) + 1e-9))
        reached_steady_state = 1.0 if rel_drift < 0.05 else 0.0
        sensitivity = self._sensitivity(model, params, t_end, t_eval, observable)

        df = pd.DataFrame({"t": t, "observable": observable})
        data_path = datasets_dir / f"mechanistic_iter{plan.iteration}.csv"
        df.to_csv(data_path, index=False)
        log_path = logs_dir / f"mechanistic_iter{plan.iteration}.json"
        log_path.write_text(
            json.dumps({"model": model, "params": params, "t_end": t_end,
                        "auc": auc, "peak": peak, "final": final,
                        "rel_drift": rel_drift, "sensitivity": sensitivity}, indent=2),
            encoding="utf-8",
        )

        metrics = [
            metric("auc", auc, note=f"area under {model} response curve"),
            metric("peak", peak, note="maximum observable value"),
            metric("final_value", final, note="value at horizon end"),
            metric("reached_steady_state", reached_steady_state,
                   note="1 if last 10% is flat (<5% drift)"),
            metric("param_sensitivity", sensitivity,
                   note="relative AUC change to +10% key parameter"),
        ]
        metrics = apply_criteria(metrics, plan.success_criteria)
        verdict, confidence = verdict_from_criteria(metrics, plan.success_criteria)
        narrative = (
            f"Integrated a real {model} ODE system over [0,{t_end}]. "
            f"AUC={auc:.3f}, peak={peak:.3f}, steady_state={'yes' if reached_steady_state else 'no'}. "
            f"Verdict: {verdict.value}."
        )

        return ValidationResult(
            run_id=run_id,
            hypothesis_statement=hypothesis,
            plan=plan,
            metrics=metrics,
            verdict=verdict,
            confidence=confidence,
            narrative=narrative,
            artifact_paths={"dataset": str(data_path), "log": str(log_path)},
        )

    # ──────────────────────── models ────────────────────────

    def _simulate(self, model, params, t_end, t_eval):
        if model == "logistic_growth":
            r = params.setdefault("r", 0.8)
            K = params.setdefault("K", 100.0)
            y0 = [params.setdefault("y0", 1.0)]

            def rhs(t, y):
                return [r * y[0] * (1 - y[0] / K)]

            sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, method="RK45")
            return sol.t, sol.y, sol.y[0]

        if model == "feedback":
            # Negative-feedback loop (digital-twin style homeostasis).
            k_prod = params.setdefault("k_prod", 1.0)
            k_deg = params.setdefault("k_deg", 0.4)
            k_fb = params.setdefault("k_fb", 0.6)
            y0 = [params.setdefault("x0", 0.0), params.setdefault("r0", 0.0)]

            def rhs(t, y):
                x, r = y
                dx = k_prod / (1 + k_fb * r) - k_deg * x
                dr = 0.5 * x - 0.3 * r
                return [dx, dr]

            sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, method="RK45")
            return sol.t, sol.y, sol.y[0]

        # default: two-compartment PK
        k_abs = params.setdefault("k_abs", 1.0)
        k_elim = params.setdefault("k_elim", 0.3)
        k12 = params.setdefault("k12", 0.5)
        k21 = params.setdefault("k21", 0.2)
        dose = params.setdefault("dose", 100.0)
        y0 = [dose, 0.0, 0.0]  # gut, central, peripheral

        def rhs(t, y):
            gut, central, periph = y
            d_gut = -k_abs * gut
            d_central = k_abs * gut - (k_elim + k12) * central + k21 * periph
            d_periph = k12 * central - k21 * periph
            return [d_gut, d_central, d_periph]

        sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, method="RK45")
        return sol.t, sol.y, sol.y[1]  # central compartment is the observable

    def _sensitivity(self, model, params, t_end, t_eval, base_observable) -> float:
        """Relative change in AUC when a key parameter is increased 10%."""
        key = {
            "logistic_growth": "r",
            "feedback": "k_fb",
            "two_compartment": "k_elim",
        }.get(model, None)
        base_auc = float(np.trapz(base_observable, t_eval)) or 1e-9
        if key is None or key not in params:
            return 0.0
        perturbed = dict(params)
        perturbed[key] = perturbed[key] * 1.1
        _, _, obs2 = self._simulate(model, perturbed, t_end, t_eval)
        new_auc = float(np.trapz(obs2, t_eval))
        return abs(new_auc - base_auc) / abs(base_auc)
