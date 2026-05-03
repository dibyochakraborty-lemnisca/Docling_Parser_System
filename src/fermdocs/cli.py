from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from sqlalchemy import create_engine

from fermdocs.dossier import build_dossier
from fermdocs.file_store.local import LocalFileStore
from fermdocs.mapping.client import dump_response_schema
from fermdocs.mapping.factory import (
    build_identity_client,
    build_mapper,
    build_narrative_extractor,
    build_segmenter,
)
from fermdocs.units.normalizer import build_default_normalizer
from fermdocs.parsing.csv_parser import CsvParser
from fermdocs.parsing.excel_parser import ExcelParser
from fermdocs.parsing.pdf_parser import DoclingPdfParser
from fermdocs.parsing.router import FormatRouter, UnsupportedFormatError
from fermdocs.pipeline import IngestionPipeline
from fermdocs.storage.repository import Repository
from fermdocs.units.converter import UnitConverter
from fermdocs.domain.golden_schema import load_schema

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_PARSE = 3
EXIT_DB = 4
EXIT_PARTIAL = 5


def _build_pipeline(
    schema_path: str | None,
    use_fake_mapper: bool,
    provider: str | None = None,
    llm_normalizer: bool | None = None,
    extract_narrative: bool | None = None,
    segment_pdfs: bool | None = None,
) -> tuple[IngestionPipeline, Repository]:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        click.echo("DATABASE_URL not set", err=True)
        sys.exit(EXIT_USAGE)
    engine = create_engine(db_url)
    repo = Repository(engine)
    repo.create_all()

    router = FormatRouter([CsvParser(), ExcelParser(), DoclingPdfParser()])
    mapper = build_mapper(provider=provider, use_fake=use_fake_mapper)
    converter = UnitConverter()
    file_store = LocalFileStore()
    schema = load_schema(schema_path) if schema_path else None

    use_llm_normalizer = (
        llm_normalizer
        if llm_normalizer is not None
        else os.environ.get("FERMDOCS_USE_LLM_NORMALIZER", "true").lower() == "true"
    )
    if use_fake_mapper:
        use_llm_normalizer = False  # offline runs stay offline
    normalizer_provider = os.environ.get("FERMDOCS_NORMALIZER_PROVIDER") or provider
    normalizer = build_default_normalizer(
        use_llm=use_llm_normalizer, provider=normalizer_provider
    )

    extract_narr = (
        extract_narrative
        if extract_narrative is not None
        else os.environ.get("FERMDOCS_EXTRACT_NARRATIVE", "true").lower() == "true"
    )
    if use_fake_mapper:
        extract_narr = False
    narrative_provider = os.environ.get("FERMDOCS_NARRATIVE_PROVIDER") or provider
    narrative_extractor = build_narrative_extractor(
        enabled=extract_narr, provider=narrative_provider
    )

    # PDF document segmenter (LLM-driven, gemini-3.1-pro-preview by default).
    # Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md
    # Default ON per design; --no-segment-pdfs or FERMDOCS_PDF_SEGMENT=false
    # disables. fake_mapper short-circuits to no segmenter (offline runs).
    use_segmenter = (
        segment_pdfs
        if segment_pdfs is not None
        else os.environ.get("FERMDOCS_PDF_SEGMENT", "true").lower() == "true"
    )
    if use_fake_mapper:
        use_segmenter = False
    segmenter_provider = (
        os.environ.get("FERMDOCS_SEGMENTER_PROVIDER")
        or ("fake" if not use_segmenter else (provider or "gemini"))
    )
    document_segmenter = build_segmenter(provider=segmenter_provider)

    return (
        IngestionPipeline(
            router, mapper, converter, repo, file_store, schema,
            normalizer, narrative_extractor,
            document_segmenter=document_segmenter,
        ),
        repo,
    )


@click.group(invoke_without_command=True)
@click.option("--print-schema", "print_schema", type=click.Choice(["dossier", "golden", "mapper"]))
@click.pass_context
def cli(ctx: click.Context, print_schema: str | None) -> None:
    if print_schema:
        click.echo(_render_schema(print_schema))
        ctx.exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(EXIT_OK)


@cli.command()
@click.option("--experiment-id", required=True)
@click.option("--files", required=True, multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--schema", "schema_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--fake-mapper", is_flag=True, help="Use deterministic FakeHeaderMapper (no LLM call).")
@click.option(
    "--provider",
    type=click.Choice(["gemini", "anthropic", "fake"], case_sensitive=False),
    default=None,
    help="LLM provider for the header mapper. Defaults to FERMDOCS_MAPPER_PROVIDER or 'gemini'.",
)
@click.option(
    "--llm-normalizer/--no-llm-normalizer",
    "llm_normalizer",
    default=None,
    help=(
        "LLM fallback for unit strings pint cannot parse. "
        "Defaults to FERMDOCS_USE_LLM_NORMALIZER (on). "
        "Use --no-llm-normalizer to disable the LLM tier (rule-based stays on)."
    ),
)
@click.option(
    "--extract-narrative/--no-extract-narrative",
    "extract_narrative",
    default=None,
    help=(
        "Extract values from narrative paragraphs (PDFs only). Uses Sonnet (~10-50x "
        "cost vs the mapper). Defaults to FERMDOCS_EXTRACT_NARRATIVE (on). "
        "Tier 1 narrative residual capture is unconditional regardless."
    ),
)
@click.option(
    "--process-manifest",
    "process_manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional YAML manifest with operator-supplied process identity. "
        "When present, the dossier records provenance=manifest and the "
        "LLM identity extractor is skipped."
    ),
)
@click.option(
    "--segment-pdfs/--no-segment-pdfs",
    "segment_pdfs",
    default=None,
    help=(
        "LLM-driven PDF document segmentation (PDF only). Splits multi-batch "
        "PDFs into per-run trajectories. Defaults to FERMDOCS_PDF_SEGMENT "
        "(on). Use --no-segment-pdfs to disable and fall back to the "
        "column-heuristic chain. CSV/Excel paths are unaffected either way."
    ),
)
@click.option(
    "--manifest-run-id",
    "manifest_run_id",
    type=str,
    default=None,
    help=(
        "Pin a single run-id for every observation in every ingested file. "
        "If the segmenter detects multiple distinct runs, a loud WARN is "
        "logged but the manifest still wins. Use when you know the file "
        "contains exactly one run."
    ),
)
def ingest(
    experiment_id: str,
    files: tuple[Path, ...],
    out: Path | None,
    schema_path: str | None,
    fake_mapper: bool,
    provider: str | None,
    llm_normalizer: bool | None,
    extract_narrative: bool | None,
    process_manifest: Path | None,
    segment_pdfs: bool | None,
    manifest_run_id: str | None,
) -> None:
    """Ingest files for an experiment, then optionally write a dossier JSON."""
    try:
        pipeline, repo = _build_pipeline(
            schema_path,
            fake_mapper,
            provider,
            llm_normalizer,
            extract_narrative,
            segment_pdfs=segment_pdfs,
        )
    except Exception as e:
        click.echo(f"db init failed: {e}", err=True)
        sys.exit(EXIT_DB)

    try:
        result = pipeline.ingest(
            experiment_id, list(files), manifest_run_id=manifest_run_id
        )
    except UnsupportedFormatError as e:
        click.echo(f"input error: {e}", err=True)
        sys.exit(EXIT_INPUT)
    except Exception as e:
        click.echo(f"parse error: {e}", err=True)
        sys.exit(EXIT_PARSE)

    click.echo(result.model_dump_json(indent=2))

    if out:
        # Manifest takes priority; LLM identity client only fires when no
        # manifest. fake_mapper short-circuits to no client (offline runs).
        identity_client = (
            None
            if (fake_mapper or process_manifest is not None)
            else build_identity_client(provider)
        )
        dossier = build_dossier(
            experiment_id,
            repo,
            manifest_path=process_manifest,
            identity_llm_client=identity_client,
        )
        out.write_text(json.dumps(dossier, indent=2, default=str))
        click.echo(f"dossier written: {out}", err=True)

    if not result.all_ok:
        sys.exit(EXIT_PARTIAL if result.any_ok else EXIT_PARSE)
    sys.exit(EXIT_OK)


@cli.command()
@click.option("--experiment-id", required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--process-manifest",
    "process_manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional YAML manifest pinning process identity (skips LLM extractor).",
)
@click.option(
    "--provider",
    type=click.Choice(["gemini", "anthropic", "fake", "none"], case_sensitive=False),
    default=None,
    help=(
        "LLM provider for the identity extractor. Defaults to "
        "FERMDOCS_IDENTITY_PROVIDER, then FERMDOCS_MAPPER_PROVIDER, then 'gemini'. "
        "Use 'fake' or 'none' to skip the LLM call (identity stays UNKNOWN)."
    ),
)
def dossier(
    experiment_id: str,
    out: Path | None,
    process_manifest: Path | None,
    provider: str | None,
) -> None:
    """Build the dossier for an already-ingested experiment."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        click.echo("DATABASE_URL not set", err=True)
        sys.exit(EXIT_USAGE)
    engine = create_engine(db_url)
    repo = Repository(engine)
    identity_client = (
        None if process_manifest is not None else build_identity_client(provider)
    )
    payload = build_dossier(
        experiment_id,
        repo,
        manifest_path=process_manifest,
        identity_llm_client=identity_client,
    )
    text = json.dumps(payload, indent=2, default=str)
    if out:
        out.write_text(text)
        click.echo(f"dossier written: {out}", err=True)
    else:
        click.echo(text)
    sys.exit(EXIT_OK)


@cli.command()
@click.option("--next", "show_next", is_flag=True, help="Show the next observation needing review.")
def review(show_next: bool) -> None:
    """Surface observations marked needs_review."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        click.echo("DATABASE_URL not set", err=True)
        sys.exit(EXIT_USAGE)
    engine = create_engine(db_url)
    repo = Repository(engine)
    if show_next:
        row = repo.next_review_observation()
        if row is None:
            click.echo("review queue empty")
        else:
            obs = repo.row_to_observation(row)
            click.echo(json.dumps(obs.to_dossier_observation(), indent=2, default=str))
    else:
        click.echo("use --next to fetch the next pending review")
    sys.exit(EXIT_OK)


def _render_schema(kind: str) -> str:
    if kind == "mapper":
        return dump_response_schema()
    if kind == "golden":
        from fermdocs.domain.models import GoldenSchema

        return json.dumps(GoldenSchema.model_json_schema(), indent=2)
    if kind == "dossier":
        return json.dumps(_DOSSIER_JSON_SCHEMA, indent=2)
    raise click.UsageError(f"unknown schema kind: {kind}")


_DOSSIER_JSON_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FermdocsDossier",
    "type": "object",
    "required": ["dossier_schema_version", "experiment", "golden_columns", "residual", "ingestion_summary"],
    "properties": {
        "dossier_schema_version": {"type": "string"},
        "experiment": {"type": "object"},
        "golden_columns": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["canonical_unit", "observations"],
                "properties": {
                    "canonical_unit": {"type": ["string", "null"]},
                    "observations": {"type": "array"},
                },
            },
        },
        "residual": {"type": "object"},
        "ingestion_summary": {"type": "object"},
    },
}


def main() -> None:
    cli()
