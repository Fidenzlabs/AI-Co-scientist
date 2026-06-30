"""Base protocol and helpers for scholarly source clients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from ..config import get_settings
from ..models import Citation


@runtime_checkable
class SourceClient(Protocol):
    """Every source client searches and returns normalized citations."""

    name: str

    def search(self, query: str, limit: int) -> list[Citation]:
        ...


def make_client(timeout: float = 20.0) -> httpx.Client:
    """HTTP client with a polite User-Agent (faster pools on Crossref/OpenAlex)."""
    settings = get_settings()
    return httpx.Client(
        timeout=timeout,
        headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        follow_redirects=True,
    )


def clean_abstract(text: str | None, limit: int = 1500) -> str | None:
    """Trim overly long abstracts to keep LLM prompts compact."""
    if not text:
        return None
    text = " ".join(text.split())
    return text[:limit]
