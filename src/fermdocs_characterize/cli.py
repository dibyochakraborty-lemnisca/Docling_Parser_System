"""CLI: read a dossier JSON, run characterization, write the output JSON.

    fermdocs-characterize <dossier.json> [--out <output.json>] [--no-validate]
                          [--bundle <out_root>] [--no-trajectory-analyzer]

The dossier may carry `_specs` (synthetic fixture format). For real ingestion
dossiers a separate setpoint table will be wired in via a SpecsProvider; for
now this CLI uses DictSpecsProvider.from_dossier.

If `--bundle` is given, a bundle directory is written under <out_root> in
addition to (or instead of) the legacy --out JSON. The bundle holds the
dossier and characterization output as a file-based handoff for the
diagnose stage.

The trajectory_analyzer (May 2026) runs after deterministic spec checks
to surface trajectory-grounded patterns via execute_python. Default ON
when a Gemini client can be built (env: FERMDOCS_CHARACTERIZE_PROVIDER /
FERMDOCS_HYPOTHESIS_PROVIDER / FERMDOCS_DIAGNOSIS_PROVIDER ladder; falls
back to stub when provider is 'fake'/'none' or GEMINI_API_KEY is unset).
Pass --no-trajectory-analyzer to force-disable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

from fermdocs import __version__ as FERMDOCS_VERSION
from fermdocs.bundle import BundleWriter
from fermdocs.domain.golden_schema import schema_version as golden_schema_version
from fermdocs_characterize.agents.llm_client import build_characterize_client
from fermdocs_characterize.agents.trajectory_analyzer import (
    build_trajectory_analyzer,
)
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.validators.output_validator import ValidationError

_log = logging.getLogger(__name__)


def _flatten_trajectories(output: CharacterizationOutput) -> list[dict]:
    """Long-format rows: one (run_id, variable, time_h, value, imputed, unit) per point.

    Diagnosis agent loads this via pd.read_csv() to do real numerical work.
    Missing values stay missing (None → empty cell on CSV write).
    """
    rows: list[dict] = []
    for t in output.trajectories:
        unit = t.unit
        for time_h, value, imputed in zip(t.time_grid, t.values, t.imputation_flags, strict=False):
            rows.append(
                {
                    "run_id": t.run_id,
                    "variable": t.variable,
                    "time_h": time_h,
                    "value": value,
                    "imputed": imputed,
                    "unit": unit,
                }
            )
    return rows


def _collect_run_ids(output: CharacterizationOutput, dossier: dict) -> list[str]:
    """Pick up run_ids from the output (preferred) or the dossier (fallback)."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for t in output.trajectories:
        if t.run_id and t.run_id not in seen_set:
            seen.append(t.run_id)
            seen_set.add(t.run_id)
    for f in output.findings:
        for rid in f.run_ids:
            if rid and rid not in seen_set:
                seen.append(rid)
                seen_set.add(rid)
    if seen:
        return seen
    # Fallback: scan dossier observations for run_id; default to a synthetic one.
    obs = dossier.get("observations") or []
    for o in obs:
        rid = (((o.get("source") or {}).get("locator") or {})).get("run_id")
        if rid and rid not in seen_set:
            seen.append(rid)
            seen_set.add(rid)
    return seen or ["RUN-UNKNOWN"]


@click.command()
@click.argument("dossier_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output JSON path. Default: stdout (unless --bundle is given).",
)
@click.option(
    "--bundle",
    "bundle_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write a bundle directory under this root (e.g. ./out). The bundle"
    " carries the dossier and characterization JSON for the diagnose stage.",
)
@click.option(
    "--no-validate",
    is_flag=True,
    default=False,
    help="Skip cross-cutting output validation (Pydantic still runs).",
)
@click.option(
    "--no-trajectory-analyzer",
    is_flag=True,
    default=False,
    help="Disable LLM-driven trajectory pattern analyzer. Default is ON"
    " when a Gemini client builds; this flag force-disables.",
)
def main(
    dossier_path: Path,
    out_path: Path | None,
    bundle_root: Path | None,
    no_validate: bool,
    no_trajectory_analyzer: bool,
) -> None:
    """Run characterization on DOSSIER_PATH."""
    dossier = json.loads(dossier_path.read_text())

    # Build trajectory_analyzer unless disabled. build_characterize_client
    # respects FERMDOCS_CHARACTERIZE_PROVIDER (with fallback ladder); it
    # returns None for 'fake'/'none', and the analyzer's stub mode then
    # makes it a no-op. So passing the analyzer in is safe even when no
    # Gemini key is set — the pipeline still produces deterministic
    # findings only.
    analyzer = None
    if not no_trajectory_analyzer:
        try:
            client = build_characterize_client()
            analyzer = build_trajectory_analyzer(client)
            if client is not None:
                click.echo(
                    "trajectory_analyzer: enabled (provider="
                    f"{client.model_name})",
                    err=True,
                )
        except Exception as exc:
            click.echo(
                f"trajectory_analyzer: disabled ({exc.__class__.__name__}: {exc})",
                err=True,
            )
            analyzer = None

    pipeline = CharacterizationPipeline(
        validate=not no_validate,
        trajectory_analyzer=analyzer,
    )
    try:
        output = pipeline.run(dossier)
    except ValidationError as e:
        click.echo(f"Validation failed: {e}", err=True)
        sys.exit(2)

    payload = output.model_dump_json(indent=2)

    if bundle_root is not None:
        run_ids = _collect_run_ids(output, dossier)
        writer = BundleWriter.create(
            bundle_root,
            run_ids=run_ids,
            golden_schema_version=golden_schema_version(),
            pipeline_version=FERMDOCS_VERSION,
            model_labels={"characterization": "deterministic/v1"},
        )
        try:
            writer.write_dossier(dossier)
            writer.write_characterization(payload)
            writer.write_observations_csv(_flatten_trajectories(output))
            if output.narrative_observations:
                writer.write_narrative_observations(
                    json.dumps(
                        [
                            n.model_dump(mode="json")
                            for n in output.narrative_observations
                        ],
                        indent=2,
                    )
                )
            bundle_path = writer.finalize()
        except Exception:
            writer.abort()
            raise
        click.echo(f"Wrote bundle {bundle_path}", err=True)

    if out_path is not None:
        out_path.write_text(payload)
        click.echo(f"Wrote {out_path}", err=True)
    elif bundle_root is None:
        # Legacy default: print to stdout when no destination flag was given.
        click.echo(payload)


if __name__ == "__main__":
    main()
