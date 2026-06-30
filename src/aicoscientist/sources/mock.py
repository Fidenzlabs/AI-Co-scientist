"""Deterministic mock source for offline runs (no network)."""

from __future__ import annotations

import hashlib

from ..models import Citation


class MockClient:
    """Generates synthetic but plausible citations from the query string."""

    name = "mock"

    def search(self, query: str, limit: int) -> list[Citation]:
        citations: list[Citation] = []
        words = [w for w in query.replace(",", " ").split() if len(w) > 2] or ["topic"]
        for i in range(min(limit, 5)):
            seed = hashlib.sha1(f"{query}:{i}".encode()).hexdigest()
            focus = words[i % len(words)]
            year = 2018 + (int(seed[:2], 16) % 8)
            citations.append(
                Citation(
                    id=Citation.make_id(self.name, seed[:12]),
                    source=self.name,
                    title=f"A study of {focus} and {words[(i + 1) % len(words)]} in {query}",
                    authors=[f"Author{seed[:4].upper()}", f"Coauthor{seed[4:8].upper()}"],
                    year=year,
                    venue="Journal of Synthetic Research",
                    url=f"https://example.org/{seed[:12]}",
                    abstract=(
                        f"This synthetic abstract discusses how {focus} relates to "
                        f"{query}. It reports that {focus} is associated with measurable "
                        f"effects and proposes a mechanism linking {focus} to "
                        f"{words[(i + 2) % len(words)]}."
                    ),
                    citation_count=int(seed[2:5], 16) % 500,
                )
            )
        return citations
