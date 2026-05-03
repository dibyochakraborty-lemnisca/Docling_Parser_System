"""CLI for the hypothesis stage.

Reads a diagnose bundle directory, runs orchestrator + 3 specialists +
synthesizer + critic + judge with real LLM agents, and writes
hypothesis_output.json + global.md to a sibling
out/hypothesis/<hypothesis_id>/ directory.

Usage:

    fermdocs-hypothesize run <bundle_dir> \\
        [--out-root DIR] \\
        [--max-turns N] [--max-tool-calls N] \\
        [--max-total-input-tokens N]

The console-script entry is `fermdocs-hypothesize`. Defaults match the
production budget: max_turns=10, max_critic_cycles_per_topic=3,
max_tool_calls_total=80, max_total_input_tokens=200000.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

# Load .env so GEMINI_API_KEY is available without sourcing.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import resume_stage, run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2


@click.group()
def cli() -> None:
    """Hypothesis stage CLI."""


@cli.command()
@click.argument(
    "bundle_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out-root",
    default="out/hypothesis",
    type=click.Path(path_type=Path),
    help="Root directory for hypothesis bundles.",
)
@click.option("--max-turns", type=int, default=10)
@click.option("--max-critic-cycles-per-topic", type=int, default=3)
@click.option("--max-tool-calls", type=int, default=80)
@click.option("--max-total-input-tokens", type=int, default=200_000)
@click.option(
    "--no-validate",
    is_flag=True,
    default=False,
    help="Skip cross-output validators (citation integrity, provenance downgrade).",
)
@click.option(
    "--hitl/--no-hitl",
    default=True,
    help="After exit, prompt to answer open questions and resume.",
)
def run(
    bundle_dir: Path,
    out_root: Path,
    max_turns: int,
    max_critic_cycles_per_topic: int,
    max_tool_calls: int,
    max_total_input_tokens: int,
    no_validate: bool,
    hitl: bool,
) -> None:
    """Run the hypothesis stage on a diagnose BUNDLE_DIR."""
    click.echo(f"loading bundle: {bundle_dir}")
    loaded = load_bundle(bundle_dir)
    click.echo(
        f"  seed_topics={len(loaded.hyp_input.seed_topics)}"
        f" findings={len(loaded.findings_pool)}"
        f" narratives={len(loaded.narratives_pool)}"
        f" trajectories={len(loaded.trajectories_pool)}"
        f" priors={len(loaded.priors_pool)}"
        f" analyses={len(loaded.analyses_pool)}"
        f" organism={loaded.hyp_input.organism!r}"
    )

    hyp_id = uuid.uuid4()
    out_dir = out_root / str(hyp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "global.md"

    budget = BudgetSnapshot(
        max_turns=max_turns,
        max_critic_cycles_per_topic=max_critic_cycles_per_topic,
        max_tool_calls_total=max_tool_calls,
        max_total_input_tokens=max_total_input_tokens,
    )

    hooks = LiveHooks(loaded)
    diagnosis_id = loaded.diagnosis.meta.diagnosis_id

    click.echo(f"running hypothesis stage (hyp_id={hyp_id})")
    result = run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        validate=not no_validate,
        now_factory=lambda: datetime.now(timezone.utc),
    )

    _persist_and_report(result, out_dir, global_md)

    if hitl:
        _maybe_hitl_loop(
            result=result,
            loaded=loaded,
            hooks=hooks,
            out_dir=out_dir,
            global_md=global_md,
            diagnosis_id=diagnosis_id,
            base_budget=budget,
            no_validate=no_validate,
        )


def _persist_and_report(result, out_dir: Path, global_md: Path) -> None:
    """Write hypothesis_output.json + print summary + token report."""
    out_path = out_dir / "hypothesis_output.json"
    out_path.write_text(
        json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
    )

    click.echo(f"\n✓ wrote {out_path}")
    click.echo(f"✓ wrote {global_md}")
    click.echo(f"\nexit reason: {result.state.exit_reason}")
    click.echo(f"final_hypotheses: {len(result.output.final_hypotheses)}")
    for h in result.output.final_hypotheses:
        click.echo(
            f"  - {h.hyp_id} (basis={h.confidence_basis.value},"
            f" conf={h.confidence:.2f}):"
        )
        click.echo(f"    {h.summary[:200]}")
    click.echo(f"rejected_hypotheses: {len(result.output.rejected_hypotheses)}")
    for r in result.output.rejected_hypotheses:
        click.echo(f"  - {r.hyp_id}: {r.rejection_reason[:120]}")
    unresolved = [q for q in result.output.open_questions if not q.resolved]
    click.echo(f"open_questions (unresolved): {len(unresolved)}")
    for q in unresolved:
        click.echo(f"  - {q.qid} (raised_by={q.raised_by}): {q.question[:160]}")

    _print_token_report(result.output.token_report)


def _maybe_hitl_loop(
    *,
    result,
    loaded,
    hooks,
    out_dir: Path,
    global_md: Path,
    diagnosis_id,
    base_budget: BudgetSnapshot,
    no_validate: bool,
) -> None:
    """If unresolved open questions exist, prompt the user one by one
    and run a resume round. Loops until no unresolved questions OR the
    user declines.
    """
    unresolved = [q for q in result.output.open_questions if not q.resolved]
    if not unresolved:
        return
    if not click.confirm(
        f"\n{len(unresolved)} open question(s). Answer them and re-run?",
        default=True,
    ):
        return

    answers: list[tuple[str, str]] = []
    for q in unresolved:
        click.echo(f"\nQ {q.qid} (raised by {q.raised_by}): {q.question}")
        ans = click.prompt(
            "  answer (empty to skip)", default="", show_default=False
        ).strip()
        if ans:
            answers.append((q.qid, ans))

    if not answers:
        click.echo("no answers provided; skipping resume.")
        return

    click.echo(f"\nresuming with {len(answers)} answer(s)...")
    resume_result = resume_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=diagnosis_id,
        answers=answers,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=base_budget,
        validate=not no_validate,
        now_factory=lambda: datetime.now(timezone.utc),
    )
    _persist_and_report(resume_result, out_dir, global_md)
    # Recurse so a multi-round dialogue is possible if the resume round
    # surfaced new open questions.
    _maybe_hitl_loop(
        result=resume_result,
        loaded=loaded,
        hooks=hooks,
        out_dir=out_dir,
        global_md=global_md,
        diagnosis_id=diagnosis_id,
        base_budget=base_budget,
        no_validate=no_validate,
    )


@cli.command()
@click.argument(
    "out_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "bundle_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--max-turns", type=int, default=10)
@click.option("--max-critic-cycles-per-topic", type=int, default=3)
@click.option("--max-tool-calls", type=int, default=80)
@click.option("--max-total-input-tokens", type=int, default=200_000)
@click.option("--no-validate", is_flag=True, default=False)
def answer(
    out_dir: Path,
    bundle_dir: Path,
    max_turns: int,
    max_critic_cycles_per_topic: int,
    max_tool_calls: int,
    max_total_input_tokens: int,
    no_validate: bool,
) -> None:
    """Resume a paused hypothesis run by answering its open questions.

    OUT_DIR is the existing run dir (e.g. out/hypothesis/<hyp_id>/) that
    contains global.md from a prior `run`. BUNDLE_DIR is the original
    diagnose bundle (we need it to reconstruct HypothesisInput).
    """
    global_md = out_dir / "global.md"
    if not global_md.exists():
        click.echo(f"no global.md in {out_dir}; nothing to resume.", err=True)
        raise SystemExit(EXIT_INPUT)

    loaded = load_bundle(bundle_dir)
    output_path = out_dir / "hypothesis_output.json"
    if not output_path.exists():
        click.echo(f"no hypothesis_output.json in {out_dir}", err=True)
        raise SystemExit(EXIT_INPUT)
    prior_output = json.loads(output_path.read_text())
    unresolved = [
        q for q in prior_output.get("open_questions", []) if not q.get("resolved")
    ]
    if not unresolved:
        click.echo("no unresolved open questions in this run.")
        return

    answers: list[tuple[str, str]] = []
    for q in unresolved:
        click.echo(f"\nQ {q['qid']} (raised by {q['raised_by']}): {q['question']}")
        ans = click.prompt(
            "  answer (empty to skip)", default="", show_default=False
        ).strip()
        if ans:
            answers.append((q["qid"], ans))

    if not answers:
        click.echo("no answers provided; aborting.")
        return

    budget = BudgetSnapshot(
        max_turns=max_turns,
        max_critic_cycles_per_topic=max_critic_cycles_per_topic,
        max_tool_calls_total=max_tool_calls,
        max_total_input_tokens=max_total_input_tokens,
    )
    hooks = LiveHooks(loaded)
    diagnosis_id = loaded.diagnosis.meta.diagnosis_id

    click.echo(f"\nresuming with {len(answers)} answer(s)...")
    result = resume_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=diagnosis_id,
        answers=answers,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        validate=not no_validate,
        now_factory=lambda: datetime.now(timezone.utc),
    )
    _persist_and_report(result, out_dir, global_md)


def _print_token_report(report) -> None:
    """Per-agent token breakdown — Stage 4 deliverable."""
    click.echo("\ntokens:")
    click.echo(
        f"  total: input={report.total_input:>7,} output={report.total_output:>6,}"
    )
    if not report.per_agent_input:
        return
    click.echo("  per-agent:")
    agents = sorted(report.per_agent_input.keys())
    for agent in agents:
        in_tok = report.per_agent_input.get(agent, 0)
        out_tok = report.per_agent_output.get(agent, 0)
        click.echo(f"    {agent:<24} input={in_tok:>7,} output={out_tok:>6,}")


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint. argv is for testing; production callers
    use the auto-discovered argv from sys.argv via click."""
    try:
        cli.main(args=argv, standalone_mode=False)
        return EXIT_OK
    except click.ClickException as e:
        e.show()
        return EXIT_USAGE
    except FileNotFoundError as e:
        click.echo(f"error: {e}", err=True)
        return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
