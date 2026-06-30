"""Interactive CLI for the AI Co-Scientist (Layers 1 & 2).

Runs Layer 1 to the Layer 2 ``interrupt``, renders the top-5 hypotheses, collects the
researcher's decision, and resumes the graph to finalize the official hypothesis.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime

from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

console = Console()


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _render_hypotheses(idea: str, hypotheses: list[dict]) -> None:
    console.rule(f"[bold]Top {len(hypotheses)} hypotheses for:[/bold] {idea}")
    for i, h in enumerate(hypotheses):
        sg = h.get("state_graph", {})
        scores = h.get("scores", {})
        support = sg.get("supporting_evidence", [])
        contra = sg.get("contradicting_evidence", [])
        refs = sg.get("references", [])

        body = Table.grid(padding=(0, 1))
        body.add_column(justify="right", style="dim")
        body.add_column()
        body.add_row("Statement", f"[bold]{h.get('statement', '')}[/bold]")
        if h.get("novelty_assessment"):
            body.add_row("Novelty", h["novelty_assessment"])
        body.add_row(
            "Scores",
            "composite=[bold]{c:.2f}[/bold]  evidence={e:.2f}  novelty={n:.2f}  "
            "consistency={k:.2f}  confidence={f:.2f}".format(
                c=scores.get("composite", 0),
                e=scores.get("evidence_quality", 0),
                n=scores.get("novelty", 0),
                k=scores.get("consistency", 0),
                f=scores.get("confidence", 0),
            ),
        )
        if support:
            body.add_row(
                "Supporting",
                "\n".join(f"+ {e['statement']}" for e in support[:3]),
            )
        if contra:
            body.add_row(
                "Contradicting",
                "\n".join(f"- {e['statement']}" for e in contra[:3]),
            )
        if sg.get("assumptions"):
            body.add_row("Assumptions", "; ".join(sg["assumptions"][:3]))
        if h.get("reasoning_trace"):
            body.add_row("Reasoning", " -> ".join(h["reasoning_trace"][:4]))
        if refs:
            body.add_row("Citations", ", ".join(refs[:6]))

        console.print(
            Panel(
                body,
                title=f"[{i + 1}] {h.get('id', '')}",
                border_style="cyan",
                expand=True,
            )
        )


def _prompt_decision(hypotheses: list[dict]) -> dict:
    id_by_index = {i + 1: h["id"] for i, h in enumerate(hypotheses)}
    console.print(
        Markdown(
            "**Choose an action:**\n"
            "- `select <n>` — accept hypothesis n as-is\n"
            "- `modify <n>` — accept n with an edited statement\n"
            "- `merge <n,m,...>` — combine several into one\n"
            "- `new` — reject all and supply a new direction\n"
            "- `quit` — abort without choosing"
        )
    )
    while True:
        try:
            raw = console.input("[bold green]decision>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            return {"action": "quit", "notes": "no input"}
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if action == "quit":
            return {"action": "quit"}
        if action == "new":
            statement = console.input("New research direction: ").strip()
            notes = console.input("Notes (optional): ").strip()
            return {"action": "new", "statement": statement, "notes": notes}
        if action in {"select", "modify"}:
            idxs = _parse_indices(arg, id_by_index)
            if not idxs:
                console.print("[red]Provide a valid hypothesis number.[/red]")
                continue
            decision = {"action": action, "selected_ids": idxs[:1]}
            if action == "modify":
                decision["statement"] = console.input("Edited statement: ").strip()
            return decision
        if action == "merge":
            idxs = _parse_indices(arg, id_by_index)
            if len(idxs) < 2:
                console.print("[red]Provide at least two numbers, e.g. 'merge 1,2'.[/red]")
                continue
            statement = console.input(
                "Merged statement (blank to auto-combine): "
            ).strip()
            return {
                "action": "merge",
                "selected_ids": idxs,
                "statement": statement or None,
            }
        console.print(f"[red]Unknown action '{action}'.[/red]")


def _parse_indices(arg: str, id_by_index: dict[int, str]) -> list[str]:
    ids: list[str] = []
    for token in arg.replace(" ", ",").split(","):
        if token.isdigit() and int(token) in id_by_index:
            ids.append(id_by_index[int(token)])
    return ids


def _auto_decision(hypotheses: list[dict], spec: str) -> dict:
    """Non-interactive decision for smoke tests, e.g. 'select:1' or 'merge:1,2'."""
    action, _, arg = spec.partition(":")
    id_by_index = {i + 1: h["id"] for i, h in enumerate(hypotheses)}
    if action == "new":
        return {"action": "new", "statement": arg or "Automated new direction."}
    if action == "quit":
        return {"action": "quit"}
    ids = _parse_indices(arg, id_by_index)
    decision = {"action": action, "selected_ids": ids}
    if action == "modify":
        decision["statement"] = "Automated modified statement."
    return decision


def run(args: argparse.Namespace) -> int:
    from .graph import compiled_graph

    settings = get_settings()
    run_id = args.run_id or _new_run_id()
    config = {"configurable": {"thread_id": run_id}}

    console.print(
        Panel.fit(
            f"[bold]AI Co-Scientist[/bold]\nidea: {args.idea}\n"
            f"run id: {run_id}\nmode: {'offline (mock)' if args.offline else 'live sources'}\n"
            f"artifacts: {settings.artifacts_path / run_id}",
            border_style="magenta",
        )
    )

    with compiled_graph() as graph:
        with console.status("[bold]Layer 1: research swarm exploring literature...[/bold]"):
            result = graph.invoke(
                {"idea": args.idea, "offline": args.offline, "run_id": run_id},
                config,
            )

        if "__interrupt__" not in result:
            console.print("[red]Pipeline finished without reaching the review step.[/red]")
            return 1

        payload = result["__interrupt__"][0].value
        hypotheses = payload.get("hypotheses", [])
        if not hypotheses:
            console.print("[red]No hypotheses were generated.[/red]")
            return 1

        for line in result.get("reasoning_trace", []):
            console.print(f"[dim]- {line}[/dim]")

        _render_hypotheses(args.idea, hypotheses)

        if args.auto:
            decision = _auto_decision(hypotheses, args.auto)
            console.print(f"[yellow]auto decision:[/yellow] {decision}")
        else:
            decision = _prompt_decision(hypotheses)

        final = graph.invoke(Command(resume=decision), config)

    official = final.get("official_hypothesis")
    console.rule("[bold]Layer 2 result[/bold]")
    if official:
        console.print(
            Panel(
                f"[bold]{official['statement']}[/bold]\n\n"
                f"origin: {official['origin']['action']}  "
                f"sources: {', '.join(official.get('source_hypothesis_ids', [])) or 'n/a'}\n"
                f"confidence: {official.get('state_graph', {}).get('confidence', 'n/a')}",
                title="Official research hypothesis",
                border_style="green",
            )
        )
        console.print(
            f"[dim]Saved to {settings.artifacts_path / run_id / 'official_hypothesis.json'}[/dim]"
        )
    else:
        console.print("[yellow]No official hypothesis was finalized (aborted).[/yellow]")

    console.print(f"[dim]All artifacts: {settings.artifacts_path / run_id}[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aicoscientist",
        description="AI Co-Scientist — Layers 1 (deep research) & 2 (human-in-the-loop).",
    )
    parser.add_argument("--idea", required=True, help="The research idea to investigate.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use mock sources and a deterministic offline pipeline (no network/LLM key).",
    )
    parser.add_argument("--run-id", default=None, help="Reuse a specific run id.")
    parser.add_argument(
        "--auto",
        default=None,
        help="Non-interactive decision for testing, e.g. 'select:1', 'merge:1,2', 'new'.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return run(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
