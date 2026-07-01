"""Drug repurposing validator (DrugPipe-inspired two-phase pipeline).

Mirrors the two-phase workflow from DrugPipe (HySonLab/DrugPipe):
  Phase 1 — generate candidate ligands from seed SMILES (RDKit mutations)
  Phase 2 — similarity retrieval vs a mini drug library + ADMET + docking proxy

Reference: Pham et al., DrugPipe, https://github.com/HySonLab/DrugPipe
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..models import ValidationPlan, ValidationResult
from .base import apply_criteria, metric, rng, verdict_from_criteria

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED
    from rdkit.Chem import rdMolDescriptors as rdmd
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:  # noqa: BLE001
    _HAVE_RDKIT = False

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_DRUGBANK_MINI = _PACKAGE_ROOT / "data" / "drugbank_mini.csv"

# Small set of common substituents for Phase 1 candidate generation.
_SUBSTITUENTS = ["F", "Cl", "C", "OC", "C(F)(F)F", "N", "C#N"]


class DrugRepurposingValidator:
    domain = "drug_repurposing"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        spec = plan.data_spec or {}
        seed_smiles = spec.get("seed_smiles", "CN(C)C(=N)N=C(N)N")  # metformin default
        n_candidates = int(spec.get("n_candidates", 12 + 4 * plan.iteration))

        if _HAVE_RDKIT:
            rows, metrics, engine = self._run_rdkit(seed_smiles, n_candidates, plan.seed)
        else:
            rows, metrics, engine = self._run_stub(seed_smiles, n_candidates, plan.seed)

        data_path = datasets_dir / f"drug_repurposing_iter{plan.iteration}.json"
        data_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        log_path = logs_dir / f"drug_repurposing_iter{plan.iteration}.json"
        log_path.write_text(
            json.dumps({
                "engine": engine,
                "phase1_seed": seed_smiles,
                "n_candidates": len(rows),
                "metrics": {m.name: m.value for m in metrics},
            }, indent=2),
            encoding="utf-8",
        )

        metrics = apply_criteria(metrics, plan.success_criteria)
        verdict, confidence = verdict_from_criteria(metrics, plan.success_criteria)
        if not _HAVE_RDKIT:
            confidence = min(confidence, 0.55)

        narrative = (
            f"DrugPipe-inspired two-phase repurposing: generated {len(rows)} candidates "
            f"from seed SMILES, ranked vs mini drug library. "
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

    # ──────────────────────── Phase 1: candidate generation ────────────────────────

    def _generate_candidates(self, seed_smiles: str, n: int, seed: int) -> list[str]:
        """Generate candidate SMILES via controlled RDKit mutations."""
        gen = rng(seed)
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            return [seed_smiles]

        candidates = {seed_smiles}
        rw = Chem.RWMol(mol)

        for _ in range(n * 3):
            if len(candidates) >= n:
                break
            try:
                sub = _SUBSTITUENTS[int(gen.integers(0, len(_SUBSTITUENTS)))]
                sub_mol = Chem.MolFromSmiles(sub)
                if sub_mol is None:
                    continue
                # Attach substituent to a random heavy atom with a free valence.
                atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetTotalNumHs(includeNeighbors=True) > 0]
                if not atoms:
                    break
                idx = int(gen.choice(atoms))
                combo = Chem.CombineMols(rw, sub_mol)
                combo_rw = Chem.RWMol(combo)
                combo_rw.AddBond(idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
                Chem.SanitizeMol(combo_rw)
                smi = Chem.MolToSmiles(combo_rw)
                if smi and Chem.MolFromSmiles(smi) is not None:
                    candidates.add(smi)
            except Exception:  # noqa: BLE001
                continue

        return list(candidates)[:n]

    # ──────────────────────── Phase 2: similarity + ADMET + docking ────────────────────────

    def _load_drug_library(self) -> list[dict]:
        if not _DRUGBANK_MINI.exists():
            return [{"name": "Metformin", "smiles": "CN(C)C(=N)N=C(N)N"}]
        with _DRUGBANK_MINI.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _run_rdkit(self, seed_smiles: str, n_candidates: int, seed: int):
        candidates = self._generate_candidates(seed_smiles, n_candidates, seed)
        library = self._load_drug_library()
        lib_fps = []
        for drug in library:
            mol = Chem.MolFromSmiles(drug["smiles"])
            if mol is not None:
                lib_fps.append((drug["name"], GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))

        rows = []
        qeds, admet_flags, sims, dock_scores = [], [], [], []

        for smi in candidates:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            tpsa = rdmd.CalcTPSA(mol)
            qed = QED.qed(mol)
            lipinski_ok = sum([mw > 500, Crippen.MolLogP(mol) > 5,
                               rdmd.CalcNumHBD(mol) > 5, rdmd.CalcNumHBA(mol) > 10]) <= 1
            tox = _tox_alerts(mol)
            admet_ok = lipinski_ok and tox == 0

            best_sim, best_drug = 0.0, ""
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for name, lib_fp in lib_fps:
                tani = DataStructs.TanimotoSimilarity(fp, lib_fp)
                if tani > best_sim:
                    best_sim, best_drug = tani, name

            dock = _docking_proxy(mw, tpsa, best_sim, seed, smi)

            qeds.append(qed)
            admet_flags.append(1.0 if admet_ok else 0.0)
            sims.append(best_sim)
            dock_scores.append(dock)
            rows.append({
                "smiles": smi, "qed": round(qed, 3),
                "admet_ok": admet_ok, "tox_alerts": tox,
                "best_similarity": round(best_sim, 3),
                "best_match_drug": best_drug,
                "docking_proxy": round(dock, 3),
            })

        rows.sort(key=lambda r: r["best_similarity"], reverse=True)
        metrics = [
            metric("n_candidates_generated", float(len(rows)),
                   note="Phase 1 generated candidates"),
            metric("max_similarity_to_known_drug", max(sims) if sims else 0.0,
                   note="Phase 2 best Tanimoto vs drug library"),
            metric("mean_qed", _mean(qeds), note="mean drug-likeness"),
            metric("admet_pass_fraction", _mean(admet_flags),
                   note="fraction passing Lipinski + no tox alerts"),
            metric("best_docking_proxy", max(dock_scores) if dock_scores else 0.0,
                   note="QVina-W stand-in (lower = better binding)"),
        ]
        return rows, metrics, "RDKit+DrugPipe-two-phase"

    def _run_stub(self, seed_smiles: str, n_candidates: int, seed: int):
        gen = rng(seed)
        rows, sims, qeds, admet_flags, docks = [], [], [], [], []
        for i in range(min(n_candidates, 8)):
            h = _hash_unit(f"{seed_smiles}:{i}:{seed}")
            sim = round(0.3 + 0.6 * h, 3)
            qed = round(0.35 + 0.5 * h, 3)
            admet = 1.0 if h > 0.4 else 0.0
            dock = round(-7.0 - 3.0 * h, 3)
            sims.append(sim)
            qeds.append(qed)
            admet_flags.append(admet)
            docks.append(dock)
            rows.append({"smiles": f"stub_{i}", "best_similarity": sim,
                         "qed": qed, "admet_ok": bool(admet), "docking_proxy": dock,
                         "note": "heuristic stub (RDKit unavailable)"})
        metrics = [
            metric("n_candidates_generated", float(len(rows))),
            metric("max_similarity_to_known_drug", max(sims) if sims else 0.0),
            metric("mean_qed", _mean(qeds)),
            metric("admet_pass_fraction", _mean(admet_flags)),
            metric("best_docking_proxy", max(docks) if docks else 0.0),
        ]
        return rows, metrics, "heuristic-stub"


def _tox_alerts(mol) -> int:
    try:
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        return len(FilterCatalog.FilterCatalog(params).GetMatches(mol))
    except Exception:  # noqa: BLE001
        return 0


def _docking_proxy(mw: float, tpsa: float, similarity: float, seed: int, smi: str) -> float:
    """Deterministic QVina-W stand-in: lower (more negative) = better predicted binding."""
    h = _hash_unit(f"{smi}:{seed}")
    # Favour drug-like MW/TPSA and similarity to known binders.
    mw_penalty = abs(mw - 350) / 350
    tpsa_penalty = max(0.0, (tpsa - 140) / 140)
    base = -6.0 - 2.5 * similarity + mw_penalty - 0.5 * tpsa_penalty
    return base - 1.5 * h


def _mean(xs) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _hash_unit(text: str) -> float:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
