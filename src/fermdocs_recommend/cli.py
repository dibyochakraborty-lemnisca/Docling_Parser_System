"""CLI for the recommendation agent.

Reads a bundle, runs the agent to evaluate brewtwin models, and
writes recommendation.json.

Usage:

    fermdocs-recommend run \
        --bundle out/bundle_1234 \
        [--provider anthropic|gemini]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fermdocs.bundle import BundleNotReady, BundleReader, BundleSchemaMismatch
from fermdocs_recommend.agent import RecommendationAgent
from fermdocs_recommend.llm_clients import build_recommend_client

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_LLM = 3


@click.group()
def cli() -> None:
    """Recommendation agent CLI."""


@cli.command()
@click.option(
    "--bundle",
    "bundle_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Bundle directory written by characterize (e.g. out/bundle_<...>/).",
)
@click.option(
    "--hypothesis-output",
    "hyp_output",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to hypothesis_output.json (the confirmed hypotheses to ground on).",
)
@click.option(
    "--output",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write recommendation.json. Defaults to <bundle>/recommend/recommendation.json.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "fake", "none"], case_sensitive=False),
    default=None,
    help="LLM provider for the ReAct loop. Defaults to 'gemini'.",
)
def run(
    bundle_dir: Path,
    hyp_output: Path | None,
    output: Path | None,
    provider: str | None,
) -> None:
    """Run the recommendation agent on a bundle."""
    try:
        reader = BundleReader(bundle_dir)
    except (BundleNotReady, BundleSchemaMismatch) as exc:
        click.echo(f"error: bundle unusable: {exc}", err=True)
        sys.exit(EXIT_INPUT)

    output = output or (reader.dir / "recommend" / "recommendation.json")

    client = build_recommend_client(provider)
    resolved = (provider or "gemini").lower()
    resolved_provider = "anthropic" if resolved == "anthropic" else "gemini"

    agent = RecommendationAgent(client=client, provider=resolved_provider)
    result = agent.recommend(bundle=reader, hypothesis_output_path=hyp_output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    click.echo(f"wrote {output} (recommended_model={result.recommended_model})")

    sys.exit(EXIT_OK if result.meta.error is None else EXIT_LLM)


main = cli

if __name__ == "__main__":
    cli()
