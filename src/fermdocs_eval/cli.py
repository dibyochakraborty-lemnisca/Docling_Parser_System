"""CLI entry: `python -m fermdocs_eval.cli <suite> ...`

Suites are implemented as separate modules under fermdocs_eval.suites and
register themselves here. v1 ships the e2 wiring first because it doesn't
depend on bundle re-ingest.
"""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """fermdocs_eval — paper evaluation suites."""


@cli.command("e1")
@click.option("--bundle", required=True, type=click.Path(exists=True))
@click.option("--question", required=True, help="Tailored user question for this bundle.")
@click.option("--process-family", default="penicillin_fedbatch", help="Process family for memory queries.")
@click.option("--out", default="eval/results/e1.jsonl")
@click.option("--no-score", is_flag=True, help="Skip LLM specificity scoring.")
def run_e1(bundle: str, question: str, process_family: str, out: str, no_score: bool) -> None:
    """E1 memory mechanism: cold then warm run on the same bundle."""
    from fermdocs_eval.suites import e1

    e1.run(
        bundle_dir=bundle, question=question, process_family=process_family,
        out_path=out, score_specificity=not no_score,
    )


@cli.command("e2")
@click.option("--out", default="eval/results/e2.jsonl")
@click.option(
    "--only",
    multiple=True,
    help="Restrict to specific fixture_ids (repeatable). Useful for dry runs.",
)
def run_e2(out: str, only: tuple[str, ...]) -> None:
    """E2 critic-axes P/R on synthetic hypotheses."""
    from fermdocs_eval.fixtures.e2_specs import SPECS
    from fermdocs_eval.suites import e2

    specs = list(SPECS)
    if only:
        wanted = set(only)
        specs = [s for s in specs if s.fixture_id in wanted]
        missing = wanted - {s.fixture_id for s in specs}
        if missing:
            raise click.ClickException(f"unknown fixture_ids: {sorted(missing)}")
    e2.run(out_path=out, specs=specs)


@cli.command("e3")
@click.option("--bundle", required=True, type=click.Path(exists=True))
@click.option("--question", required=True, help="Tailored user question for this bundle.")
@click.option("--out", default="eval/results/e3.jsonl")
def run_e3(bundle: str, question: str, out: str) -> None:
    """E3 case study: pipeline vs single-shot Gemini baseline."""
    from fermdocs_eval.suites import e3

    e3.run(bundle_dir=bundle, question=question, out_path=out)


if __name__ == "__main__":
    cli()
