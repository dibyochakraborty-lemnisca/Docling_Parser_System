"""CLI: read a dossier JSON, run characterization, write the output JSON.

    fermdocs-characterize <dossier.json> [--out <output.json>] [--no-validate]
                          [--bundle <out_root>]

The dossier may carry `_specs` (synthetic fixture format). For real ingestion
dossiers a separate setpoint table will be wired in via a SpecsProvider; for
now this CLI uses DictSpecsProvider.from_dossier.

If `--bundle` is given, a bundle directory is written under <out_root> in
addition to (or instead of) the legacy --out JSON. The bundle holds the
dossier and characterization output as a file-based handoff for the
diagnose stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fermdocs import __version__ as FERMDOCS_VERSION
from fermdocs.bundle import BundleWriter
from fermdocs.domain.golden_schema import schema_version as golden_schema_version
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.validators.output_validator import ValidationError


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
def main(
    dossier_path: Path,
    out_path: Path | None,
    bundle_root: Path | None,
    no_validate: bool,
) -> None:
    """Run characterization on DOSSIER_PATH."""
    dossier = json.loads(dossier_path.read_text())
    pipeline = CharacterizationPipeline(validate=not no_validate)
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
