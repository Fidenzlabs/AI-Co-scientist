# AS-ALD Inhibitor / Precursor Selection Criteria

**This file is optional.** By default (`PRIORS_SOURCE=auto`) the pipeline runs with **no
manual input**: Layer 1 mines inhibitor adsorption energies and reactivity from the
retrieved papers into the knowledge graph (`kg_candidates.json`), and the Layer-3
**selection agent** (ADR-005, Deliverable #2) uses those literature-grounded priors,
falling back to the built-in defaults below.

Edit this file only if you want to **override** the machine-mined priors with your own
domain expertise — add molecules, or change `dE_ngs`/`dE_gs`. To make your edits win over
the KG-mined values, set `PRIORS_SOURCE=manual` in `.env`. Precedence:

- `PRIORS_SOURCE=auto`   → KG-mined  >  this file  >  built-in defaults  *(default)*
- `PRIORS_SOURCE=manual` → this file >  KG-mined   >  built-in defaults

Across refinement iterations the agent walks down the ranked list, so you can "let it run
against different iterations" to explore alternative candidates automatically.

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
6. **Site-matched screening** (Kim et al. 2026) — precursors prefer specific reactive sites
   (e.g. BDEAS → −OH; TMA/DMAI → −OH + −O− bridge). Inhibitors with explicit
   `site_reactivity` (DMATMS, ETS, or KG-mined entries) are scored by how well their
   reactive sites cover the precursor's preferred sites, in addition to the criteria above.

## Candidate library (literature-grounded priors)

The selection agent intersects this library with candidates found in the knowledge graph,
then ranks by the criteria above. `dE_ngs` / `dE_gs` are literature/xTB adsorption-energy
priors (eV) used by the Tier-0 protocol and as the calibration anchor for Tier-1 MLIP.
For Kim et al. 2026 silylamine/chlorosilane inhibitors, optional `site_reactivity` carries
per-site-type ΔEr and Ea (eV); the engine uses full site-resolved blocking for those entries
while carboxylic acids keep the legacy terminal-site curve.

```json
{
  "inhibitors": {
    "acetic acid":               {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high",   "removability": "high"},
    "pivalic acid":              {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high",   "removability": "high"},
    "ethylbutyric acid":         {"dE_ngs": -0.98, "dE_gs": -0.24, "functional_group": "carboxylic acid", "volatility": "medium", "removability": "high"},
    "methanesulfonic acid":      {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid",   "volatility": "medium", "removability": "medium"},
    "aniline":                   {"dE_ngs": -0.90, "dE_gs": -0.57, "functional_group": "aromatic amine",  "volatility": "high",   "removability": "high"},
    "octadecylphosphonic acid":  {"dE_ngs": -1.30, "dE_gs": -0.30, "functional_group": "phosphonic acid", "volatility": "low",    "removability": "low"},
    "DMATMS":                    {"dE_ngs": -0.70, "dE_gs": -0.18, "functional_group": "silyl amine",     "volatility": "high",   "removability": "medium",
                                  "site_reactivity": {"SiO2": {"OH": {"deltaEr_eV": -0.85, "Ea_eV": 0.48}, "O_bridge": {"deltaEr_eV": 0.64, "Ea_eV": 1.50}},
                                                     "SiN":  {"NH2": {"deltaEr_eV": -0.80, "Ea_eV": 1.34}, "NH_bridge": {"deltaEr_eV": -0.70, "Ea_eV": 1.54}}}},
    "ETS":                       {"dE_ngs": -0.95, "dE_gs": -0.30, "functional_group": "chlorosilane",    "volatility": "high",   "removability": "medium",
                                  "site_reactivity": {"SiO2": {"OH": {"deltaEr_eV": -0.30, "Ea_eV": 1.10}, "O_bridge": {"deltaEr_eV": 0.74, "Ea_eV": 1.46}},
                                                     "SiN":  {"NH2": {"deltaEr_eV": -0.95, "Ea_eV": 0.79}, "NH_bridge": {"deltaEr_eV": -0.85, "Ea_eV": 0.80}}}}
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
