"""Typer CLI for TriageMCP: triage batches, run the eval, serve MCP, emit samples.

Annotations are intentionally evaluated at runtime (no ``from __future__ import annotations``)
because Typer resolves parameter types at import time.
"""

import asyncio
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import TypeAdapter, ValidationError
from rich.console import Console

from triagemcp.agent.voting import VotingTriager
from triagemcp.config import Settings
from triagemcp.datasets import load_sample_alerts
from triagemcp.eval.experiments import PROMPT_VARIANTS, run_experiment
from triagemcp.eval.harness import eval_reference_clock, run_eval, write_headline_to_readme
from triagemcp.eval.metrics import EvalReport
from triagemcp.models import Alert, LabeledAlert, TriageOutcome
from triagemcp.pipeline import triage_batch
from triagemcp.report import (
    eval_report_table,
    outcomes_table,
    outcomes_to_json,
    outcomes_to_markdown,
)
from triagemcp.server import build_runtime, serve_stdio
from triagemcp.tools.mitre import load_mitre_techniques

app = typer.Typer(
    name="triagemcp",
    help="Agentic security-alert triage engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_ALERTS = TypeAdapter(list[Alert])
_LABELED = TypeAdapter(list[LabeledAlert])


class OutputFormat(StrEnum):
    table = "table"
    json = "json"
    markdown = "markdown"


def _abort(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _load_settings(model: str | None) -> Settings:
    try:
        settings = Settings()
    except ValidationError:
        _abort(
            "Configuration error: ANTHROPIC_API_KEY is not set. "
            "Set it in your environment or a .env file (see .env.example)."
        )
    if model is not None:
        settings.model = model
    return settings


def _load_alerts(file: Path | None) -> list[Alert]:
    """Load alerts from a file (plain alerts or labeled alerts), or the bundled samples."""
    if file is None:
        return [item.alert for item in load_sample_alerts()]
    data = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], dict) and "alert" in data[0]:
        return [item.alert for item in _LABELED.validate_python(data)]
    return _ALERTS.validate_python(data)


def _emit_outcomes(outcomes: list[TriageOutcome], fmt: OutputFormat, out: Path | None) -> None:
    if fmt is OutputFormat.table:
        table = outcomes_table(outcomes)
        if out is None:
            console.print(table)
        else:
            with out.open("w", encoding="utf-8") as handle:
                Console(file=handle, width=120).print(table)
            console.print(f"Wrote table to {out}")
        return

    text = (
        outcomes_to_json(outcomes) if fmt is OutputFormat.json else outcomes_to_markdown(outcomes)
    )
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        console.print(f"Wrote {fmt.value} report to {out}")


async def _triage(alerts: list[Alert], settings: Settings, concurrency: int) -> list[TriageOutcome]:
    async with build_runtime(settings) as triager:
        return await triage_batch(alerts, triager, concurrency=concurrency)


async def _evaluate(settings: Settings, concurrency: int, vote_samples: int | None) -> EvalReport:
    labeled = load_sample_alerts()
    tactic_by_id = {technique.id: technique.tactic for technique in load_mitre_techniques()}
    clock = eval_reference_clock(labeled)
    samples = settings.vote_samples if vote_samples is None else vote_samples
    temperature = 0.7 if samples > 1 else 0.0  # voting needs diversity; a single sample stays at 0
    async with build_runtime(settings, temperature=temperature, clock=clock) as agent:
        triager = VotingTriager(agent, samples) if samples > 1 else agent
        return await run_eval(labeled, triager, concurrency=concurrency, tactic_by_id=tactic_by_id)


@app.command()
def triage(
    file: Annotated[
        Path | None,
        typer.Argument(help="JSON file of alerts or labeled alerts; defaults to the sample set."),
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format.")
    ] = OutputFormat.table,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the report here instead of stdout.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", "-c", min=1, help="Max concurrent triages.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override the Anthropic model.")
    ] = None,
) -> None:
    """Triage a batch of alerts with the live agent and emit a validated report."""
    alerts = _load_alerts(file)
    if not alerts:
        _abort("No alerts to triage.")
    settings = _load_settings(model)
    outcomes = asyncio.run(_triage(alerts, settings, concurrency or settings.concurrency))
    _emit_outcomes(outcomes, output_format, out)


@app.command(name="eval")
def evaluate(
    update_readme: Annotated[
        bool, typer.Option("--update-readme", help="Write the headline accuracy into a README.")
    ] = False,
    readme: Annotated[Path, typer.Option(help="README path used with --update-readme.")] = Path(
        "README.md"
    ),
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c", min=1)] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    vote_samples: Annotated[
        int | None,
        typer.Option("--vote-samples", min=1, help="Self-consistency samples to majority-vote."),
    ] = None,
) -> None:
    """Run the agent over the labeled sample set and report accuracy."""
    settings = _load_settings(model)
    report = asyncio.run(_evaluate(settings, concurrency or settings.concurrency, vote_samples))
    console.print(eval_report_table(report))
    console.print(report.summary_line())
    if update_readme:
        write_headline_to_readme(report, readme)
        console.print(f"Updated {readme}")


@app.command()
def serve(
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Start the MCP server over stdio (exposes the triage_alert tool)."""
    settings = _load_settings(model)
    asyncio.run(serve_stdio(settings))


@app.command()
def sample(
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the samples here instead of stdout.")
    ] = None,
) -> None:
    """Emit the bundled labeled sample alerts as JSON."""
    labeled = load_sample_alerts()
    payload = json.dumps([item.model_dump(mode="json") for item in labeled], indent=2)
    if out is None:
        typer.echo(payload)
    else:
        out.write_text(payload, encoding="utf-8")
        console.print(f"Wrote {len(labeled)} sample alerts to {out}")


@app.command()
def experiment(
    variant: Annotated[
        str, typer.Option("--variant", "-v", help="Prompt variant to evaluate.")
    ] = "baseline",
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c", min=1)] = None,
) -> None:
    """Run the eval for one prompt variant (for the accuracy showcase)."""
    if variant not in PROMPT_VARIANTS:
        _abort(f"Unknown variant {variant!r}. Choose from: {', '.join(sorted(PROMPT_VARIANTS))}.")
    settings = _load_settings(model)
    report = asyncio.run(
        run_experiment(
            variant,
            settings=settings,
            model=settings.model,
            concurrency=concurrency or settings.concurrency,
        )
    )
    console.print(f"[bold]Variant:[/] {variant}  [bold]model:[/] {settings.model}")
    console.print(eval_report_table(report))
    console.print(report.summary_line())


if __name__ == "__main__":
    app()
