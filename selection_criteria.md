# AS-ALD Inhibitor / Precursor Selection Criteria

Human-editable criteria the Layer-3 **selection agent** (ADR-005, Deliverable #2) uses to
rank inhibitor/precursor candidates retrieved from the Layer-1 knowledge graph. Edit the
prose to steer chemistry priors, and edit the `candidates` block to add molecules or
update literature/xTB adsorption-energy priors.

## Ranking criteria (qualitative)

1. **Differential adsorption** — strong chemisorption on the non-growth surface (NGS,
   `dE_ngs` well below −0.7 eV) and only weak physisorption on the growth surface (GS,
   `dE_gs` above −0.3 eV). This differential is the selectivity driver.
2. **Volatility / vapor pressure** — the inhibitor must be dosable in the gas phase; higher
   volatility is preferred for uniform passivation.
3. **Functional-group ↔ site compatibility** — carboxylic/phosphonic acids for −OH/−NH
   sites; N-aromatics for metal NGS.
4. **Steric size** — large enough to block precursor adsorption once chemisorbed.
5. **Post-deposition removability** — the passivant must be removable without damaging the
   film (favors small carboxylic acids over long-chain phosphonic acids).

## Candidate library (literature-grounded priors)

The selection agent intersects this library with candidates found in the knowledge graph,
then ranks by the criteria above. `dE_ngs` / `dE_gs` are literature/xTB adsorption-energy
priors (eV) used by the Tier-0 protocol and as the calibration anchor for Tier-1 MLIP.

```json
{
  "inhibitors": {
    "acetic acid":               {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high",   "removability": "high"},
    "pivalic acid":              {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high",   "removability": "high"},
    "ethylbutyric acid":         {"dE_ngs": -0.98, "dE_gs": -0.24, "functional_group": "carboxylic acid", "volatility": "medium", "removability": "high"},
    "methanesulfonic acid":      {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid",   "volatility": "medium", "removability": "medium"},
    "aniline":                   {"dE_ngs": -0.90, "dE_gs": -0.15, "functional_group": "aromatic amine",  "volatility": "high",   "removability": "high"},
    "octadecylphosphonic acid":  {"dE_ngs": -1.30, "dE_gs": -0.30, "functional_group": "phosphonic acid", "volatility": "low",    "removability": "low"},
    "DMATMS":                    {"dE_ngs": -0.70, "dE_gs": -0.18, "functional_group": "silyl amine",     "volatility": "high",   "removability": "medium"}
  },
  "precursors": {
    "BDEAS": {"target_film": "SiOx"},
    "DIPAS": {"target_film": "SiOx"},
    "HCDS":  {"target_film": "SiOx"},
    "TDMAT": {"target_film": "TiN"},
    "TMA":   {"target_film": "Al2O3"},
    "DMAI":  {"target_film": "Al2O3"}
  }
}
```
