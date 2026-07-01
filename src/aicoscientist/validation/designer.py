"""Experiment designer agent (ReAct).

Turns an approved hypothesis into a concrete, runnable in-silico ``ValidationPlan``:
it reasons about the hypothesis (ReAct), selects the validation domain/engine, and
emits parameters, a synthetic-data spec, and quantitative success criteria. An optional
Reflection critique is incorporated when refining on later loop iterations. Without an
LLM key it falls back to a deterministic, keyword-driven design so offline runs work.
"""

from __future__ import annotations

import logging

from ..llm import structured_call
from ..models import (
    OfficialHypothesis,
    Reflection,
    SuccessCriterion,
    ValidationPlan,
)

logger = logging.getLogger(__name__)

VALID_DOMAINS = {
    "statistical", "cheminformatics", "mechanistic", "protein",
    "drug_repurposing", "structure_based_design",
}

_DESIGN_SYSTEM = (
    "You are the experiment-design agent of an AI co-scientist performing in-silico "
    "validation. Follow a ReAct process: first reason briefly about how to test the "
    "hypothesis computationally (record short steps in reasoning_trace), then emit a "
    "concrete plan. Choose exactly one domain from: statistical, cheminformatics, "
    "mechanistic, drug_repurposing, structure_based_design, protein. Provide method, "
    "parameters, a data_spec describing the synthetic dataset/model to generate, "
    "quantitative success_criteria (metric, operator, threshold), assumptions, and a "
    "random seed. Criteria must be objectively checkable from the engine's metrics."
)

# Keyword cues for the deterministic fallback domain router.
_REPURPOSING_CUES = (
    "repurpos", "virtual screen", "drugbank", "drug repurposing", "reposition",
    "existing drug", "approved drug", "metformin", "off-label",
)
_SBDD_CUES = (
    "pocket", "sbdd", "structure-based", "structure based", "diffusion ligand",
    "protease", "binding pocket", "equivariant", "de novo design", "ligand design",
    "hiv protease", "protein-ligand",
)
_CHEM_CUES = (
    "drug", "molecule", "compound", "inhibitor", "ligand", "binding", "smiles",
    "pharmacophore", "small-molecule", "small molecule", "scaffold", "agonist",
    "antagonist", "aspirin", "kinase", "receptor binding",
)
_PROTEIN_CUES = (
    "protein structure", "fold", "folding", "secondary structure", "amino acid",
    "peptide", "enzyme structure", "structure prediction", "sequence",
)
_MECH_CUES = (
    "pathway", "dynamics", "kinetic", "concentration", "oscillat", "feedback",
    "growth", "pharmacokinetic", "pk/pd", "pk-pd", "systems biology", "flux",
    "steady state", "signaling", "metabolic", "digital twin",
)


class ExperimentDesigner:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def design(
        self,
        official: OfficialHypothesis,
        concept_names: list[str] | None = None,
        prior_critique: Reflection | None = None,
        iteration: int = 0,
    ) -> ValidationPlan:
        concept_names = concept_names or []
        if self.offline:
            plan = self._design_heuristic(official, iteration)
        else:
            plan = self._design_llm(official, concept_names, prior_critique, iteration)
        return self._ensure_defaults(plan, official, iteration)

    # ──────────────────────── LLM (ReAct) ────────────────────────

    def _design_llm(
        self,
        official: OfficialHypothesis,
        concept_names: list[str],
        prior_critique: Reflection | None,
        iteration: int,
    ) -> ValidationPlan:
        concepts = ", ".join(concept_names[:25]) or "(none)"
        critique_block = ""
        if prior_critique and prior_critique.decision == "refine":
            adjustments = "; ".join(prior_critique.suggested_adjustments)
            critique_block = (
                f"\n\nA previous validation attempt was critiqued. Refine the design "
                f"accordingly.\nCritique: {prior_critique.critique}\n"
                f"Suggested adjustments: {adjustments}"
            )
        user = (
            f"Hypothesis: {official.statement}\n"
            f"Stated confidence: {official.state_graph.confidence}\n"
            f"Related knowledge-graph concepts: {concepts}\n"
            f"Iteration: {iteration}{critique_block}\n\n"
            "Design the in-silico validation plan."
        )
        try:
            plan = structured_call(ValidationPlan, _DESIGN_SYSTEM, user)
            plan.iteration = iteration
            return plan
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM experiment design failed: %s", exc)
            return self._design_heuristic(official, iteration)

    # ──────────────────────── deterministic fallback ────────────────────────

    def _design_heuristic(
        self, official: OfficialHypothesis, iteration: int
    ) -> ValidationPlan:
        domain = self._route_domain(official.statement.lower())
        return self._build_for_domain(domain, official, iteration)

    def _build_for_domain(
        self, domain: str, official: OfficialHypothesis, iteration: int
    ) -> ValidationPlan:
        conf = official.state_graph.confidence or 0.5
        seed = 42 + iteration

        plan = ValidationPlan(
            domain=domain,
            seed=seed,
            iteration=iteration,
            reasoning_trace=[
                f"Classified hypothesis into '{domain}' domain via keyword cues.",
                f"Calibrated assumed effect from stated confidence {conf:.2f}.",
                "Selected quantitative success criteria for the chosen engine.",
            ],
            assumptions=[
                "Synthetic data/model is calibrated to literature-implied effect sizes.",
                "Assumptions are logged; results are reproducible from the seed.",
            ],
        )

        # On refinement, strengthen the test (more data / stricter generation).
        boost = 1.0 + 0.5 * iteration
        if domain == "statistical":
            effect = round(0.2 + 0.6 * conf, 3)
            plan.method = "Synthetic two-group comparison with regression + t-test"
            plan.data_spec = {
                "design": "two_group",
                "n": int(120 * boost),
                "effect_size": effect,
                "noise": 1.0,
            }
            plan.success_criteria = [
                SuccessCriterion(metric="p_value", operator="<", threshold=0.05,
                                 description="effect is statistically significant"),
                SuccessCriterion(metric="cohens_d", operator=">=", threshold=0.3,
                                 description="effect size is at least small-to-medium"),
            ]
        elif domain == "cheminformatics":
            plan.method = "RDKit drug-likeness, ADMET/toxicity alerts, drug-target similarity"
            plan.data_spec = {
                "smiles": [
                    "CN(C)C(=N)N=C(N)N",          # metformin
                    "CC(=O)OC1=CC=CC=C1C(=O)O",   # aspirin
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
                ],
                "reference_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            }
            plan.success_criteria = [
                SuccessCriterion(metric="mean_qed", operator=">=", threshold=0.4,
                                 description="acceptable mean drug-likeness (QED)"),
                SuccessCriterion(metric="lipinski_pass_fraction", operator=">=", threshold=0.5,
                                 description="majority pass Lipinski rule-of-five"),
                SuccessCriterion(metric="max_tanimoto", operator=">=", threshold=0.3,
                                 description="at least one candidate resembles the target ligand"),
            ]
        elif domain == "mechanistic":
            plan.method = "ODE simulation of pathway/PK-PD dynamics (scipy solve_ivp)"
            plan.data_spec = {
                "model": "two_compartment",
                "params": {"k_abs": 1.0, "k_elim": 0.3, "k12": 0.5, "k21": 0.2, "dose": 100.0},
                "t_end": 24.0,
                "n_points": 200,
            }
            plan.success_criteria = [
                SuccessCriterion(metric="auc", operator=">", threshold=10.0,
                                 description="non-trivial systemic exposure"),
                SuccessCriterion(metric="reached_steady_state", operator=">=", threshold=1.0,
                                 description="system stabilizes within the horizon"),
            ]
        elif domain == "drug_repurposing":
            plan.method = (
                "DrugPipe two-phase: RDKit candidate generation + similarity retrieval "
                "+ ADMET + docking proxy"
            )
            plan.data_spec = {
                "seed_smiles": "CN(C)C(=N)N=C(N)N",  # metformin
                "n_candidates": int(12 * boost),
            }
            plan.assumptions = [
                "Phase 1 generates candidates via RDKit mutations from seed SMILES.",
                "Phase 2 ranks vs bundled mini drug library (DrugPipe-inspired).",
                "Docking proxy is a QVina-W stand-in; real QVina-W is out of scope.",
            ]
            plan.success_criteria = [
                SuccessCriterion(metric="max_similarity_to_known_drug", operator=">=",
                                 threshold=0.25,
                                 description="candidate resembles a known approved drug"),
                SuccessCriterion(metric="admet_pass_fraction", operator=">=", threshold=0.3,
                                 description="some candidates pass ADMET filters"),
                SuccessCriterion(metric="best_docking_proxy", operator="<", threshold=-5.0,
                                 description="predicted binding affinity proxy"),
            ]
        elif domain in ("structure_based_design", "protein"):
            plan.domain = "structure_based_design"
            plan.method = "DiffSBDD SE(3)-equivariant diffusion SBDD (stub backend)"
            plan.data_spec = {
                "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
                "n_samples": int(5 * boost),
            }
            plan.assumptions = [
                "Stub backend produces deterministic metrics.",
                "Real SBDD via DiffSBDD Colab when GPU/models available.",
            ]
            plan.success_criteria = [
                SuccessCriterion(metric="ligand_validity", operator=">=", threshold=0.6,
                                 description="majority of generated ligands are valid"),
                SuccessCriterion(metric="qed", operator=">=", threshold=0.35,
                                 description="acceptable drug-likeness"),
                SuccessCriterion(metric="binding_proxy", operator="<", threshold=-6.0,
                                 description="predicted binding affinity"),
            ]
        else:  # fallback statistical
            plan.domain = "statistical"
            effect = round(0.2 + 0.6 * conf, 3)
            plan.method = "Synthetic two-group comparison with regression + t-test"
            plan.data_spec = {
                "design": "two_group",
                "n": int(120 * boost),
                "effect_size": effect,
                "noise": 1.0,
            }
            plan.success_criteria = [
                SuccessCriterion(metric="p_value", operator="<", threshold=0.05,
                                 description="effect is statistically significant"),
                SuccessCriterion(metric="cohens_d", operator=">=", threshold=0.3,
                                 description="effect size is at least small-to-medium"),
            ]
        return plan

    @staticmethod
    def _route_domain(text: str) -> str:
        if any(cue in text for cue in _SBDD_CUES):
            return "structure_based_design"
        if any(cue in text for cue in _REPURPOSING_CUES):
            return "drug_repurposing"
        if any(cue in text for cue in _PROTEIN_CUES):
            return "structure_based_design"
        if any(cue in text for cue in _CHEM_CUES):
            return "cheminformatics"
        if any(cue in text for cue in _MECH_CUES):
            return "mechanistic"
        return "statistical"

    # ──────────────────────── normalization ────────────────────────

    def _ensure_defaults(
        self, plan: ValidationPlan, official: OfficialHypothesis, iteration: int
    ) -> ValidationPlan:
        """Guarantee a runnable plan even if the LLM returned a sparse one."""
        if plan.domain not in VALID_DOMAINS:
            plan.domain = "statistical"
        plan.iteration = iteration
        if not plan.seed:
            plan.seed = 42 + iteration

        # If the LLM omitted criteria or required data_spec keys, backfill from the
        # deterministic design for the same domain.
        needs_spec = not plan.data_spec or not plan.success_criteria
        if needs_spec:
            fb = self._build_for_domain(plan.domain, official, iteration)
            plan.data_spec = plan.data_spec or fb.data_spec
            plan.success_criteria = plan.success_criteria or fb.success_criteria
            plan.method = plan.method or fb.method
        return plan
