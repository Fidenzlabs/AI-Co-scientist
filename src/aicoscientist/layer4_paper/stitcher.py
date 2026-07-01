"""Layer 4 - Agentic LaTeX paper stitcher (ADR-007, capstone).

Assembles a reproducible manuscript from the Layer-1->3 artifacts of a completed run:

* Hypothesis + results  <- ``asald_results.json``
* Surface methods table  <- ``surface_fidelity.json``
* Real citations         <- ``citation_repository.json`` (actual DOIs)

Per-section writer agents emit LaTeX fragments (numbers only from artifacts), a figure
agent renders the selectivity plot, and the compiler agent builds the PDF (degrading to a
``.tex`` source when no TeX toolchain is present). Nothing is invented.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from . import sections
from .compiler import compile_pdf
from .figures import selectivity_figure

logger = logging.getLogger(__name__)

_TEMPLATE = Path(__file__).parent / "template.tex"


class PaperDataError(RuntimeError):
    """Raised when the artifacts required to stitch a manuscript are missing."""


@dataclass
class PaperResult:
    run_id: str
    tex_path: Path
    pdf_path: Path | None
    figure_path: Path | None
    verdict: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stitch_paper(run_id: str) -> PaperResult:
    settings = get_settings()
    run_dir = settings.artifacts_path / run_id
    rich_path = run_dir / "asald_results.json"
    if not rich_path.exists():
        raise PaperDataError(
            f"No asald_results.json for run '{run_id}'. Run Layer 3 validation first."
        )
    rich = _load_json(rich_path)

    fidelity_path = run_dir / "surface_fidelity.json"
    fidelity = _load_json(fidelity_path) if fidelity_path.exists() else {}

    citations = []
    cite_path = run_dir / "citation_repository.json"
    if cite_path.exists():
        citations = _load_json(cite_path).get("citations", [])

    paper_dir = run_dir / "manuscript"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Figure agent (real numbers only).
    figure_path = selectivity_figure(rich, paper_dir / "selectivity.png")
    figure_block = ""
    if figure_path is not None:
        figure_block = (
            "\\begin{figure}[h]\\centering\n"
            f"\\includegraphics[width=0.8\\linewidth]{{{figure_path.name}}}\n"
            "\\caption{Area-selectivity vs oxide thickness with the target line and the "
            "surface-ensemble band.}\n\\end{figure}"
        )

    # Section writer agents.
    filled = _TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__TITLE__": sections.title(rich),
        "__DATE__": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "__ABSTRACT__": sections.abstract(rich),
        "__INTRODUCTION__": sections.introduction(rich),
        "__METHODS__": sections.methods(rich, fidelity),
        "__RESULTS__": sections.results(rich),
        "__FIGURE__": figure_block,
        "__DISCUSSION__": sections.discussion(rich),
        "__CONCLUSION__": sections.conclusion(rich),
        "__BIBLIOGRAPHY__": sections.bibliography(citations),
    }
    for key, val in replacements.items():
        filled = filled.replace(key, val)

    tex_path = paper_dir / "manuscript.tex"
    tex_path.write_text(filled, encoding="utf-8")
    logger.info("wrote manuscript source to %s", tex_path)

    pdf_path = compile_pdf(tex_path)

    return PaperResult(
        run_id=run_id,
        tex_path=tex_path,
        pdf_path=pdf_path,
        figure_path=figure_path,
        verdict=rich.get("verdict", "inconclusive"),
    )
