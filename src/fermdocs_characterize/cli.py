"""CLI: read a dossier JSON, run characterization, write the output JSON.

    fermdocs-characterize <dossier.json> [--out <output.json>] [--no-validate]

The dossier may carry `_specs` (synthetic fixture format). For real ingestion
dossiers a separate setpoint table will be wired in via a SpecsProvider; for
now this CLI uses DictSpecsProvider.from_dossier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.validators.output_validator import ValidationError


@click.command()
@click.argument("dossier_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output JSON path. Default: stdout.",
)
@click.option(
    "--no-validate",
    is_flag=True,
    default=False,
    help="Skip cross-cutting output validation (Pydantic still runs).",
)
def main(dossier_path: Path, out_path: Path | None, no_validate: bool) -> None:
    """Run characterization on DOSSIER_PATH."""
    dossier = json.loads(dossier_path.read_text())
    pipeline = CharacterizationPipeline(validate=not no_validate)
    try:
        output = pipeline.run(dossier)
    except ValidationError as e:
        click.echo(f"Validation failed: {e}", err=True)
        sys.exit(2)

    payload = output.model_dump_json(indent=2)
    if out_path is None:
        click.echo(payload)
    else:
        out_path.write_text(payload)
        click.echo(f"Wrote {out_path}", err=True)


if __name__ == "__main__":
    main()
