"""Deterministic mock source for offline runs (no network).

Emits AS-ALD-flavored synthetic citations so ``--offline`` still produces a
surface-chemistry knowledge graph (inhibitors, precursors, surfaces, mechanisms).
"""

from __future__ import annotations

import hashlib

from ..models import Citation

# Surface-chemistry vocabulary the offline KG should be built from.
_INHIBITORS = ["acetic acid", "pivalic acid", "methanesulfonic acid", "aniline",
               "octadecylphosphonic acid", "DMATMS"]
_PRECURSORS = ["BDEAS", "DIPAS", "HCDS", "TDMAT", "TMA"]
_SURFACES = ["a-SiO2 growth surface", "a-SiN non-growth surface",
             "silanol sites", "amine -NH sites"]
_MECHS = ["chemisorption on the non-growth surface",
          "physisorption on the growth surface",
          "nucleation delay", "differential adsorption energy",
          "precursor half-reaction barrier"]


class MockClient:
    """Generates synthetic but plausible AS-ALD citations from the query string."""

    name = "mock"

    def search(self, query: str, limit: int) -> list[Citation]:
        citations: list[Citation] = []
        for i in range(min(limit, 5)):
            seed = hashlib.sha1(f"{query}:{i}".encode()).hexdigest()
            h = int(seed[:6], 16)
            inh = _INHIBITORS[h % len(_INHIBITORS)]
            prec = _PRECURSORS[(h >> 3) % len(_PRECURSORS)]
            surf = _SURFACES[(h >> 5) % len(_SURFACES)]
            mech = _MECHS[(h >> 7) % len(_MECHS)]
            year = 2018 + (int(seed[:2], 16) % 8)
            citations.append(
                Citation(
                    id=Citation.make_id(self.name, seed[:12]),
                    source=self.name,
                    title=(f"Area-selective deposition: {inh} inhibitor with {prec} "
                           f"precursor on {surf}"),
                    authors=[f"Author{seed[:4].upper()}", f"Coauthor{seed[4:8].upper()}"],
                    year=year,
                    venue="Journal of Synthetic Surface Chemistry",
                    url=f"https://example.org/{seed[:12]}",
                    abstract=(
                        f"This synthetic abstract reports that the inhibitor {inh} "
                        f"passivates the {surf} via {mech}, while the {prec} precursor "
                        f"grows the target film on the growth surface. Selectivity is "
                        f"driven by {mech}; measured selectivity approaches the target "
                        f"at 10 nm oxide thickness."
                    ),
                    citation_count=int(seed[2:5], 16) % 500,
                )
            )
        return citations
