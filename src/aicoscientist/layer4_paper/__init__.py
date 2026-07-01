"""Layer 4 - Agentic LaTeX paper stitcher (ADR-007)."""

from .stitcher import PaperDataError, PaperResult, stitch_paper

__all__ = ["PaperDataError", "PaperResult", "stitch_paper"]
