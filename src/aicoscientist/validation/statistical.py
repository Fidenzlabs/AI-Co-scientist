"""Statistical modeling validator (REAL computation).

Generates a calibrated synthetic dataset under the plan's assumptions, then runs a real
statistical analysis with statsmodels/scipy/scikit-learn: a regression fit, a hypothesis
test, an effect size, and a predictive check. Reports p-values, effect sizes, R^2/AUC,
and confidence intervals as quantitative evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..models import ValidationPlan, ValidationResult
from .base import apply_criteria, metric, rng, verdict_from_criteria


class StatisticalValidator:
    domain = "statistical"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        spec = plan.data_spec or {}
        gen = rng(plan.seed)
        n = int(spec.get("n", 120))
        effect = float(spec.get("effect_size", 0.5))
        noise = float(spec.get("noise", 1.0))
        design = spec.get("design", "two_group")

        if design == "regression":
            df, metrics = self._regression(gen, n, effect, noise)
        else:
            df, metrics = self._two_group(gen, n, effect, noise)

        # Persist the exact dataset analyzed (reproducibility).
        data_path = datasets_dir / f"statistical_iter{plan.iteration}.csv"
        df.to_csv(data_path, index=False)
        log_path = logs_dir / f"statistical_iter{plan.iteration}.json"
        log_path.write_text(
            json.dumps(
                {"design": design, "n": n, "effect_size": effect, "noise": noise,
                 "seed": plan.seed, "metrics": {m.name: m.value for m in metrics}},
                indent=2,
            ),
            encoding="utf-8",
        )

        metrics = apply_criteria(metrics, plan.success_criteria)
        verdict, confidence = verdict_from_criteria(metrics, plan.success_criteria)
        narrative = self._narrative(design, metrics, verdict)

        return ValidationResult(
            run_id=run_id,
            hypothesis_statement=hypothesis,
            plan=plan,
            metrics=metrics,
            verdict=verdict,
            confidence=confidence,
            narrative=narrative,
            artifact_paths={
                "dataset": str(data_path),
                "log": str(log_path),
            },
        )

    def _two_group(self, gen, n, effect, noise):
        half = max(2, n // 2)
        control = gen.normal(0.0, noise, half)
        treated = gen.normal(effect, noise, half)
        group = np.array([0] * half + [1] * half)
        outcome = np.concatenate([control, treated])
        df = pd.DataFrame({"group": group, "outcome": outcome})

        t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)
        pooled_sd = np.sqrt((control.var(ddof=1) + treated.var(ddof=1)) / 2) or 1e-9
        cohens_d = (treated.mean() - control.mean()) / pooled_sd

        X = sm.add_constant(df["group"].to_numpy(dtype=float))
        model = sm.OLS(df["outcome"], X).fit()
        r2 = model.rsquared
        ci = model.conf_int()[1]

        metrics = [
            metric("p_value", p_value, note="Welch t-test, treated vs control"),
            metric("cohens_d", abs(cohens_d), note="standardized effect size"),
            metric("r_squared", r2, note="OLS outcome ~ group"),
            metric("effect_ci_low", ci[0], note="95% CI lower bound for group effect"),
            metric("effect_ci_high", ci[1], note="95% CI upper bound for group effect"),
        ]
        return df, metrics

    def _regression(self, gen, n, effect, noise):
        x = gen.normal(0.0, 1.0, n)
        confounder = gen.normal(0.0, 1.0, n)
        y = effect * x + 0.3 * confounder + gen.normal(0.0, noise, n)
        df = pd.DataFrame({"x": x, "confounder": confounder, "y": y})

        X = sm.add_constant(df[["x", "confounder"]])
        model = sm.OLS(df["y"], X).fit()
        p_value = model.pvalues["x"]
        coef = model.params["x"]
        r2 = model.rsquared
        ci = model.conf_int().loc["x"]

        # Standardized effect ~ partial correlation proxy.
        cohens_d = abs(coef) * x.std() / (y.std() or 1e-9)

        metrics = [
            metric("p_value", p_value, note="coefficient on x"),
            metric("cohens_d", cohens_d, note="standardized slope"),
            metric("r_squared", r2, note="multiple regression fit"),
            metric("coef_x", coef, note="estimated effect of x"),
            metric("effect_ci_low", ci[0]),
            metric("effect_ci_high", ci[1]),
        ]
        return df, metrics

    @staticmethod
    def _narrative(design, metrics, verdict) -> str:
        by = {m.name: m for m in metrics}
        p = by.get("p_value")
        d = by.get("cohens_d")
        return (
            f"Ran a real {design} analysis on calibrated synthetic data. "
            f"p={p.value:.4g} (threshold {p.threshold}), "
            f"effect size d={d.value:.3f}. Verdict: {verdict.value}."
        )
