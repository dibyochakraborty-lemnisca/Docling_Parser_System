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
@click.option("--out", default="eval/results/e1.jsonl")
def run_e1(bundle: str, question: str, out: str) -> None:
    """E1 memory mechanism: cold then warm run on the same bundle."""
    from fermdocs_eval.suites import e1

    e1.run(bundle_dir=bundle, question=question, out_path=out)


@cli.command("e2")
@click.option("--out", default="eval/results/e2.jsonl")
def run_e2(out: str) -> None:
    """E2 critic-axes P/R on synthetic hypotheses."""
    from fermdocs_eval.suites import e2

    e2.run(out_path=out)


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
