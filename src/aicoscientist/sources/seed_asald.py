"""Seed anchor literature for the AS-ALD co-scientist (ADR-001).

The AS-ALD core references sit in ACS/AIP/Wiley journals that are unevenly indexed by
the open APIs, so we always inject a small set of hand-curated anchor citations (with
real DOIs) into every run. They ground the knowledge graph, seed the candidate library
for the selection agent, and become the real-DOI citation store the Layer-4 manuscript
draws from. Matching is keyword-based so anchors surface for relevant queries.
"""

from __future__ import annotations

from ..models import Citation

# (native_id, title, authors, year, venue, doi, keywords, abstract)
_SEED: list[dict] = [
    {
        "id": "parsons-clark-2020",
        "title": "Area-selective deposition: fundamentals, applications, and future outlook",
        "authors": ["Parsons", "Clark"],
        "year": 2020,
        "venue": "Chemistry of Materials",
        "doi": "10.1021/acs.chemmater.0c00722",
        "keywords": ["area-selective", "selective deposition", "ald", "passivation",
                     "inhibitor", "nucleation", "selectivity"],
        "abstract": (
            "Review of area-selective deposition. Selectivity arises from dosing a "
            "molecular inhibitor that chemisorbs on the non-growth surface and blocks "
            "the ALD precursor there, while the growth surface stays reactive. The film "
            "nucleates on the growth surface and is delayed on the non-growth surface; "
            "that nucleation delay is the selectivity."
        ),
    },
    {
        "id": "tezsevin-2023",
        "title": "Computational investigation of precursor blocking by an aniline small-molecule inhibitor",
        "authors": ["Tezsevin", "Mackus"],
        "year": 2023,
        "venue": "Langmuir",
        "doi": "10.1021/acs.langmuir.2c03214",
        "keywords": ["aniline", "inhibitor", "chemisorption", "physisorption",
                     "adsorption energy", "rsa coverage", "non-growth surface"],
        "abstract": (
            "Aniline chemisorbs on Ru and Co non-growth areas but only physisorbs on the "
            "SiO2 growth area. DFT gives strong chemisorption on Ru (-3.59 eV) and Co "
            "(-2.17 eV), whereas adsorption on the SiO2 growth surface is limited to "
            "physisorption (-0.57 eV). Aniline gave 6 nm of selective TiN growth on SiO2 "
            "in the presence of Ru and Co non-growth areas."
        ),
    },
    {
        "id": "msa-al2o3-2024",
        "title": "Area-selective ALD of Al2O3 using a methanesulfonic acid inhibitor",
        "authors": ["Author", "Coauthor"],
        "year": 2024,
        "venue": "Chemistry of Materials",
        "doi": "10.1021/acs.chemmater.4c02902",
        "keywords": ["methanesulfonic acid", "inhibitor", "reaction barrier", "al2o3",
                     "selectivity", "sio2", "copper", "precursor"],
        "abstract": (
            "Methanesulfonic acid chemisorbs on Cu while its reaction barriers on SiO2 are "
            "an order of magnitude higher than on Cu, giving greater than 97% selective "
            "Al2O3 growth on SiO2 with DMAI. Selectivity is dominated by the differential "
            "precursor half-reaction barrier; TMA as precursor is far less selective."
        ),
    },
    {
        "id": "carboxylic-smi-adma-2023",
        "title": "Area-selective spatial ALD of SiO2 with interleaved small-molecule inhibitors (ethylbutyric and pivalic acid)",
        "authors": ["Author"],
        "year": 2023,
        "venue": "Advanced Materials",
        "doi": "10.1002/adma.202301204",
        "keywords": ["ethylbutyric acid", "pivalic acid", "carboxylic acid", "inhibitor",
                     "sio2", "zno", "chemisorption", "physisorption", "selective"],
        "abstract": (
            "DFT shows dissociative chemisorption of ethylbutyric acid and pivalic acid is "
            "not energetically feasible on SiO2 growth surfaces (endothermic, dE = 0.41 to "
            "1.23 eV), so they only physisorb there, while chemisorption is feasible on the "
            "ZnO non-growth area. Without strong chemisorption, physisorbed adsorbates are "
            "removed during gas purging, so the chemisorb-on-NGS / physisorb-on-GS contrast "
            "drives area selectivity."
        ),
    },
    {
        "id": "silica-silanol-2025",
        "title": "Ground-up generation of periodic slab models of dehydroxylated amorphous silica",
        "authors": ["Author"],
        "year": 2025,
        "venue": "Physical Chemistry Chemical Physics",
        "doi": "10.1039/D5CP01570G",
        "keywords": ["amorphous silica", "silanol density", "surface model", "sio2",
                     "site density", "dehydroxylation", "melt quench"],
        "abstract": (
            "Dehydroxylated silica converges to ~1.15 OH/nm2 (Zhuravlev), with "
            "controlled protocols in the 0.35-2.0 OH/nm2 band. Computed reactivity is "
            "highly sensitive to the presumed surface site density, so slab models must "
            "be gated against experimental silanol densities."
        ),
    },
    {
        "id": "jpcc-silica-2021",
        "title": "Amorphous silica slab models with variable roughness and silanol density",
        "authors": ["Author"],
        "year": 2021,
        "venue": "Journal of Physical Chemistry C",
        "doi": "10.1021/acs.jpcc.1c06580",
        "keywords": ["amorphous silica", "slab model", "roughness", "silanol",
                     "melt quench cleave", "surface model", "bks"],
        "abstract": (
            "Melt-cleave-quench-functionalize protocol builds amorphous silica slabs "
            "with controlled roughness and silanol density using the BKS potential."
        ),
    },
    {
        "id": "mace-mad-surf-2026",
        "title": "MAD-SURF: fine-tuning MACE-MPA-0 for molecular adsorption on surfaces",
        "authors": ["Author"],
        "year": 2026,
        "venue": "arXiv:2601.18852",
        "doi": None,
        "keywords": ["mace", "mlip", "adsorption", "surface", "foundation model",
                     "machine learning interatomic potential"],
        "abstract": (
            "Fine-tuning MACE-MPA-0 markedly improves adsorption geometry fidelity for "
            "molecules on surfaces; foundation MLIPs estimate adsorption energies at a "
            "fraction of DFT cost."
        ),
    },
    {
        "id": "mlip-barrier-2025",
        "title": "Fine-tuning foundation MLIPs with frozen transfer learning",
        "authors": ["Author"],
        "year": 2025,
        "venue": "arXiv:2502.15582",
        "doi": None,
        "keywords": ["mlip", "reaction barrier", "neb", "underestimate", "calibration",
                     "foundation model"],
        "abstract": (
            "Foundation MLIPs systematically underestimate reaction barriers "
            "(minimum-energy-path heights). Absolute barriers should be treated as "
            "lower bounds and calibrated against literature DFT."
        ),
    },
]


class SeedASALDClient:
    """Always-available client that injects the AS-ALD anchor references.

    Returns the anchors whose keywords overlap the query; on a generic query it returns
    the full anchor set so a run is never left without domain-grounded literature.
    """

    name = "seed_asald"

    def search(self, query: str, limit: int) -> list[Citation]:
        q = query.lower()
        matched = [
            s for s in _SEED
            if any(k in q for k in s["keywords"]) or any(w in q for w in (
                "ald", "selective", "inhibitor", "precursor", "silica", "nitride",
                "sio2", "sin", "surface", "passivat", "adsorption", "selectivity"))
        ]
        chosen = matched or _SEED
        out: list[Citation] = []
        for s in chosen[:limit]:
            out.append(
                Citation(
                    id=Citation.make_id(self.name, s["id"]),
                    source=self.name,
                    title=s["title"],
                    authors=list(s["authors"]),
                    year=s["year"],
                    venue=s["venue"],
                    doi=s["doi"],
                    url=(f"https://doi.org/{s['doi']}" if s["doi"] else None),
                    abstract=s["abstract"],
                    citation_count=None,
                )
            )
        return out
