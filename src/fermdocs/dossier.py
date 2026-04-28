from __future__ import annotations

from collections import defaultdict
from typing import Any

from fermdocs.domain.golden_schema import load_schema
from fermdocs.storage.repository import Repository

DOSSIER_SCHEMA_VERSION = "1.0"


def build_dossier(experiment_id: str, repository: Repository) -> dict[str, Any]:
    schema = load_schema()
    schema_index = schema.by_name()
    experiment = repository.fetch_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"experiment {experiment_id!r} not found")

    files = repository.fetch_files(experiment_id)
    obs_rows = repository.fetch_active_observations(experiment_id)
    residuals = repository.fetch_residuals(experiment_id)

    by_column: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obs_rows:
        obs = repository.row_to_observation(row)
        by_column[obs.column_name].append(obs.to_dossier_observation())

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

    return {
        "dossier_schema_version": DOSSIER_SCHEMA_VERSION,
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "uploaded_by": experiment.uploaded_by,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "status": experiment.status,
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
        },
    }
