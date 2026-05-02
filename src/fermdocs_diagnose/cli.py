"""CLI for the diagnosis agent.

Reads a CharacterizationOutput JSON + dossier JSON, runs the agent, and
optionally writes markdown sidecars.

Usage:

    fermdocs-diagnose run \\
        --dossier path/to/dossier.json \\
        --characterization path/to/characterization.json \\
        --output path/to/diagnosis.json \\
        [--emit-markdown DIR] \\
        [--provider anthropic|gemini]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_diagnose.agent import DiagnosisAgent
from fermdocs_diagnose.llm_clients import build_diagnosis_client
from fermdocs_diagnose.renderers import (
    render_analysis_md,
    render_diagnosis_md,
    render_failures_md,
    render_questions_md,
    render_trends_md,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_LLM = 3


@click.group()
def cli() -> None:
    """Diagnosis agent CLI."""


@cli.command()
@click.option(
    "--dossier",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to dossier JSON.",
)
@click.option(
    "--characterization",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to upstream CharacterizationOutput JSON.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write DiagnosisOutput JSON.",
)
@click.option(
    "--emit-markdown",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory to write Failures.md, Trends.md, Analysis.md, Questions.md,"
        " Diagnosis.md. Off by default — structured JSON is the contract."
    ),
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "fake", "none"], case_sensitive=False),
    default=None,
    help=(
        "LLM provider for the diagnosis ReAct loop. Defaults to "
        "FERMDOCS_DIAGNOSIS_PROVIDER, then FERMDOCS_MAPPER_PROVIDER, then 'gemini'. "
        "Use 'fake' or 'none' to skip the LLM call (meta.error path)."
    ),
)
def run(
    dossier: Path,
    characterization: Path,
    output: Path,
    emit_markdown: Path | None,
    provider: str | None,
) -> None:
    """Run the diagnosis agent on a (dossier, characterization) pair."""
    try:
        dossier_data = json.loads(dossier.read_text())
        char_data = json.loads(characterization.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        click.echo(f"error: failed to read inputs: {exc}", err=True)
        sys.exit(EXIT_INPUT)

    char_output = CharacterizationOutput.model_validate(char_data)

    client = build_diagnosis_client(provider)
    # Resolve the provider label that ends up on meta.provider. The factory
    # may have used a fallback chain; pass through for audit.
    resolved_provider = (provider or "gemini").lower()
    if resolved_provider not in ("anthropic", "gemini"):
        resolved_provider = "gemini"  # meta.provider only accepts those two
    agent = DiagnosisAgent(client=client, provider=resolved_provider)
    result = agent.diagnose(dossier_data, char_output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.model_dump(mode="json"), indent=2))
    click.echo(f"wrote {output}")

    if emit_markdown is not None:
        emit_markdown.mkdir(parents=True, exist_ok=True)
        (emit_markdown / "Failures.md").write_text(render_failures_md(result))
        (emit_markdown / "Trends.md").write_text(render_trends_md(result))
        (emit_markdown / "Analysis.md").write_text(render_analysis_md(result))
        (emit_markdown / "Questions.md").write_text(render_questions_md(result))
        (emit_markdown / "Diagnosis.md").write_text(render_diagnosis_md(result))
        click.echo(f"wrote markdown sidecars to {emit_markdown}")

    sys.exit(EXIT_OK if result.meta.error is None else EXIT_LLM)


main = cli


if __name__ == "__main__":
    cli()
