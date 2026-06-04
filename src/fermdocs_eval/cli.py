"""CLI entry: `python -m fermdocs_eval.cli <suite> ...`

Suites:
  - ablation: E3a ablation studies (full vs component-removed variants)
  - headtohead: legacy agent vs single-shot baseline (deprecated)
"""

from __future__ import annotations

import sys

# Disable Python 3.11+'s 4300-digit limit on int-from-string conversion.
# Gemini occasionally returns responses containing very long numeric
# strings, and the google-genai SDK's internal json.loads then crashes
# with ValueError. The CVE this limit guards against (quadratic-time
# attacks on huge integers) doesn't apply here — we control the input
# distribution and there are no untrusted callers. Scoped to the eval
# process only.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

import click


@click.group()
def cli() -> None:
    """fermdocs_eval — paper evaluation."""


@cli.command("headtohead")
@click.option("--bundle", required=True, type=click.Path(exists=True))
@click.option(
    "--questions",
    default="eval/questions.json",
    help="JSON file mapping qid -> question text.",
)
@click.option("--out", default="eval/results/headtohead.jsonl")
@click.option(
    "--only",
    multiple=True,
    help="Restrict to specific qids (repeatable). Useful for dry runs.",
)
def run_headtohead(bundle: str, questions: str, out: str, only: tuple[str, ...]) -> None:
    """Run the agent vs single-shot baseline on the bundle's questions."""
    import json
    from pathlib import Path

    from fermdocs_eval.suites import headtohead

    q_all = json.loads(Path(questions).read_text())
    if only:
        wanted = set(only)
        q = {k: v for k, v in q_all.items() if k in wanted}
        missing = wanted - set(q.keys())
        if missing:
            raise click.ClickException(f"unknown qids: {sorted(missing)}")
    else:
        q = q_all

    # Write filtered questions to a temp file so the suite reads the right set.
    if only:
        tmp = Path(out).parent / "_questions_subset.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(q, indent=2))
        questions_path = str(tmp)
    else:
        questions_path = questions

    headtohead.run(
        bundle_dir=bundle,
        questions_path=questions_path,
        out_path=out,
    )


@cli.command("ablation")
@click.option("--bundle", required=True, type=click.Path(exists=True))
@click.option(
    "--questions",
    default="eval/questions.json",
    help="JSON file mapping qid -> question text.",
)
@click.option("--out", default="eval/results/ablations.jsonl")
@click.option(
    "--only-q",
    multiple=True,
    help="Restrict to specific qids (repeatable).",
)
@click.option(
    "--only-config",
    multiple=True,
    help="Restrict to specific config names (repeatable).",
)
def run_ablation(
    bundle: str,
    questions: str,
    out: str,
    only_q: tuple[str, ...],
    only_config: tuple[str, ...],
) -> None:
    """Run E3a ablation matrix on the bundle."""
    from fermdocs_eval.ablation_configs import ALL_CONFIGS, ABLATION_QUESTIONS
    from fermdocs_eval.suites import ablations

    qids = tuple(only_q) if only_q else ABLATION_QUESTIONS
    if only_config:
        wanted = set(only_config)
        cfgs = tuple(c for c in ALL_CONFIGS if c.name in wanted)
        missing = wanted - {c.name for c in cfgs}
        if missing:
            raise click.ClickException(f"unknown configs: {sorted(missing)}")
    else:
        cfgs = ALL_CONFIGS

    ablations.run(
        bundle_dir=bundle,
        questions_path=questions,
        out_path=out,
        questions=qids,
        configs=cfgs,
    )


if __name__ == "__main__":
    cli()
