"""Scholarly source clients and aggregator."""

from .aggregator import SourceAggregator
from .arxiv import ArxivClient
from .crossref import CrossrefClient
from .mock import MockClient
from .openalex import OpenAlexClient
from .pubmed import PubMedClient
from .semantic_scholar import SemanticScholarClient

__all__ = [
    "SourceAggregator",
    "ArxivClient",
    "CrossrefClient",
    "MockClient",
    "OpenAlexClient",
    "PubMedClient",
    "SemanticScholarClient",
]
