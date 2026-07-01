"""Structure-based drug design validator (DiffSBDD-inspired).

Uses an SE(3)-equivariant diffusion backend interface to generate ligands conditioned
on a protein pocket. The default stub produces deterministic metrics; real generation
is available via the DiffSBDD Colab notebook or a local DiffSBDD install (DIFFSBDD_PATH).

Reference: Schneuing et al., arXiv:2210.13695
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import ValidationPlan, ValidationResult
from .backends.diffsbdd import get_backend
from .base import apply_criteria, metric, verdict_from_criteria


class StructureBasedDesignValidator:
    domain = "structure_based_design"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        spec = plan.data_spec or {}
        pocket_pdb = spec.get("pocket_pdb")
        sequence = spec.get("sequence")
        n_samples = int(spec.get("n_samples", 5 + 2 * plan.iteration))

        backend = get_backend()
        result = backend.generate(pocket_pdb, sequence, n_samples, plan.seed)

        data_path = datasets_dir / f"sbdd_iter{plan.iteration}.json"
        data_path.write_text(
            json.dumps({
                "ligands": result.ligand_smiles,
                "backend": result.backend,
                "note": result.note,
            }, indent=2),
            encoding="utf-8",
        )
        log_path = logs_dir / f"sbdd_iter{plan.iteration}.json"
        log_path.write_text(
            json.dumps({
                "backend": result.backend,
                "pocket_pdb": pocket_pdb,
                "sequence_len": len(sequence) if sequence else 0,
                "n_generated": result.n_generated,
                "metrics": {
                    "ligand_validity": result.ligand_validity,
                    "qed": result.qed,
                    "sa_score": result.sa_score,
                    "binding_proxy": result.binding_proxy,
                },
            }, indent=2),
            encoding="utf-8",
        )

        metrics = [
            metric("ligand_validity", result.ligand_validity,
                   note="fraction of valid generated ligands"),
            metric("qed", result.qed, note="drug-likeness of best ligand"),
            metric("sa_score", result.sa_score, unit="1-10",
                   note="synthetic accessibility (lower = easier)"),
            metric("binding_proxy", result.binding_proxy, unit="kcal/mol",
                   note="predicted binding affinity proxy (lower = better)"),
            metric("n_generated", float(result.n_generated),
                   note="number of ligands generated"),
        ]
        metrics = apply_criteria(metrics, plan.success_criteria)
        verdict, confidence = verdict_from_criteria(metrics, plan.success_criteria)
        if result.backend.endswith("stub"):
            confidence = min(confidence, 0.55)

        narrative = (
            f"Structure-based design via {result.backend}: generated "
            f"{result.n_generated} ligand(s). QED={result.qed:.3f}, "
            f"binding_proxy={result.binding_proxy:.2f}. {result.note} "
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
