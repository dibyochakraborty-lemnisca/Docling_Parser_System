from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from fermdocs.domain.golden_schema import load_schema
from fermdocs.domain.models import (
    IdentityProvenance,
    NarrativeBlock,
    NarrativeBlockType,
    ObservedFacts,
    ProcessIdentity,
    RegisteredProcess,
    ScaleInfo,
)
from fermdocs.mapping.identity_extractor import (
    IdentityExtractor,
    IdentityLLMClient,
)
from fermdocs.mapping.process_registry import cached_registry
from fermdocs.storage.repository import Repository

DOSSIER_SCHEMA_VERSION = "1.1"

_log = logging.getLogger(__name__)


def load_process_manifest(path: str | Path) -> ProcessIdentity:
    """Load an operator-supplied identity manifest YAML into a ProcessIdentity.

    The manifest format is intentionally flat -- operators don't need to know
    about the observed/registered split. We populate both layers from the
    same source and force provenance=MANIFEST on both, so a manifest can
    never disguise itself as LLM-extracted.

    Accepted YAML keys:
      organism, product, process_family (or process_family_hint),
      scale: {volume_l, vessel_type},
      process_id (registry id, optional),
      confidence (default 1.0), rationale.

    Examples are in /plans for the format.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path!s} must be a YAML mapping")

    confidence = float(data.get("confidence", 1.0))
    confidence = max(0.0, min(1.0, confidence))

    scale_data = data.get("scale")
    scale: ScaleInfo | None = None
    if isinstance(scale_data, dict):
        scale = ScaleInfo(
            volume_l=scale_data.get("volume_l"),
            vessel_type=scale_data.get("vessel_type"),
        )
    elif "scale_volume_l" in data or "vessel_type" in data:
        scale = ScaleInfo(
            volume_l=data.get("scale_volume_l"),
            vessel_type=data.get("vessel_type"),
        )

    family_hint = data.get("process_family_hint") or data.get("process_family")

    observed = ObservedFacts(
        organism=data.get("organism"),
        product=data.get("product"),
        process_family_hint=family_hint,
        scale=scale,
        confidence=confidence,
        provenance=IdentityProvenance.MANIFEST,
        rationale=data.get("rationale"),
    )

    registered = RegisteredProcess(
        process_id=data.get("process_id"),
        confidence=confidence,
        provenance=IdentityProvenance.MANIFEST,
        rationale=data.get("rationale"),
    )

    return ProcessIdentity(observed=observed, registered=registered)


def _residual_narrative_blocks(residuals: list) -> list[NarrativeBlock]:
    """Reconstruct NarrativeBlock objects from residuals.

    Each residual payload's `narrative` list is `list[dict]` (the
    JSON-serialized form of NarrativeBlock written at ingestion). We
    rehydrate so the identity extractor receives the same shape it
    receives from the live ingestion pipeline.
    """
    blocks: list[NarrativeBlock] = []
    for r in residuals:
        for entry in r.payload.get("narrative", []):
            if not isinstance(entry, dict):
                continue
            try:
                blocks.append(NarrativeBlock.model_validate(entry))
            except Exception:
                # malformed historical entry -> skip rather than fail the build
                continue
    return blocks


def _present_variables(by_column: dict[str, Any]) -> set[str]:
    return {k for k, v in by_column.items() if v}


def _resolve_identity(
    *,
    manifest_path: str | Path | None,
    llm_client: IdentityLLMClient | None,
    narrative_blocks: list[NarrativeBlock],
    present_variables: set[str],
) -> ProcessIdentity:
    """Priority chain: manifest > LLM extractor > UNKNOWN."""
    if manifest_path is not None:
        return load_process_manifest(manifest_path)

    extractor = IdentityExtractor(cached_registry(), llm_client)
    return extractor.extract(
        narrative_blocks, present_variables=present_variables
    )


def build_dossier(
    experiment_id: str,
    repository: Repository,
    *,
    manifest_path: str | Path | None = None,
    identity_llm_client: IdentityLLMClient | None = None,
) -> dict[str, Any]:
    schema = load_schema()
    schema_index = schema.by_name()
    experiment = repository.fetch_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"experiment {experiment_id!r} not found")

    files = repository.fetch_files(experiment_id)
    obs_rows = repository.fetch_active_observations(experiment_id)
    residuals = repository.fetch_residuals(experiment_id)

    by_column: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stale_versions: set[str] = set()
    for row in obs_rows:
        obs = repository.row_to_observation(row)
        if obs.schema_version is not None and obs.schema_version != schema.version:
            stale_versions.add(obs.schema_version)
        by_column[obs.column_name].append(obs.to_dossier_observation())

    if stale_versions:
        _log.warning(
            "experiment %s has observations from older schema versions %s; "
            "current schema is %s. Mappings and unit semantics may have shifted.",
            experiment_id,
            sorted(stale_versions),
            schema.version,
        )

    golden_columns: dict[str, Any] = {}
    for col_name, observations in by_column.items():
        canonical_unit = schema_index[col_name].canonical_unit if col_name in schema_index else None
        golden_columns[col_name] = {
            "canonical_unit": canonical_unit,
            "observations": observations,
        }

    total_obs = len(obs_rows)
    high_conf = sum(
        1
        for r in obs_rows
        if r.mapping_confidence is not None and float(r.mapping_confidence) >= 0.85
    )
    needs_review = sum(1 for r in obs_rows if r.needs_review)
    fell_to_residual = sum(
        len(r.payload.get("tables_partial", [])) + len(r.payload.get("tables_unmapped", []))
        for r in residuals
    )
    columns_with_obs = len(by_column)
    columns_total = len(schema.columns)
    coverage_percent = (
        round(columns_with_obs * 100 / columns_total) if columns_total else 0
    )
    files_failed = sum(1 for f in files if f.parse_status == "failed")

    narrative_kept = 0
    narrative_blocks_total = 0
    for r in residuals:
        narrative_blocks_total += len(r.payload.get("narrative", []))
    for row in obs_rows:
        if row.source_locator and row.source_locator.get("section") == "narrative":
            narrative_kept += 1

    identity = _resolve_identity(
        manifest_path=manifest_path,
        llm_client=identity_llm_client,
        narrative_blocks=_residual_narrative_blocks(residuals),
        present_variables=_present_variables(by_column),
    )

    return {
        "dossier_schema_version": DOSSIER_SCHEMA_VERSION,
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "uploaded_by": experiment.uploaded_by,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "status": experiment.status,
            "process": identity.model_dump(mode="json"),
            "source_files": [
                {
                    "file_id": str(f.file_id),
                    "filename": f.filename,
                    "sha256": f.sha256,
                    "page_count": f.page_count,
                    "parse_status": f.parse_status,
                    "parse_error": f.parse_error,
                }
                for f in files
            ],
        },
        "golden_columns": golden_columns,
        "residual": {
            "summary": {
                "residual_records": len(residuals),
                "tables_unmapped": sum(
                    len(r.payload.get("tables_unmapped", [])) for r in residuals
                ),
                "tables_partial": sum(
                    len(r.payload.get("tables_partial", [])) for r in residuals
                ),
            },
            "records": [
                {
                    "residual_id": str(r.residual_id),
                    "file_id": str(r.file_id),
                    "extractor_version": r.extractor_version,
                    "payload": r.payload,
                }
                for r in residuals
            ],
        },
        "ingestion_summary": {
            "total_observations": total_obs,
            "high_confidence": high_conf,
            "needs_review": needs_review,
            "fell_to_residual": fell_to_residual,
            "golden_coverage_percent": coverage_percent,
            "files_failed_to_parse": files_failed,
            "schema_version": schema.version,
            "stale_schema_versions": sorted(stale_versions),
            "narrative_blocks_captured": narrative_blocks_total,
            "narrative_observations": narrative_kept,
        },
    }
