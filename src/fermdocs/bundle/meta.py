"""Bundle metadata schema and version-mismatch errors.

`meta.json` is the only cross-stage handshake. Its presence in the bundle
directory is the signal that the bundle is fully written; absence means a
half-written bundle that BundleReader must refuse.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BUNDLE_SCHEMA_VERSION = "1.0"
"""Bundle layout version. Exact-match required across writer and reader."""


class BundleSchemaMismatch(Exception):
    """Bundle was written under a different bundle_schema_version. Hard fail."""


class GoldenSchemaMajorMismatch(Exception):
    """Bundle was written under a different MAJOR golden_schema_version. Hard fail."""


class BundleMeta(BaseModel):
    """Cross-stage handshake. Last file written before atomic rename.

    Fields:
        bundle_schema_version: this layout's version. Reader requires exact match.
        golden_schema_version: golden_schema.yaml version active at write time.
            Reader: major mismatch fails, minor mismatch warns.
        pipeline_version: writer's package version (or git short sha if dev).
        created_at: UTC, ISO 8601.
        bundle_id: equals the bundle directory's basename suffix.
        run_ids: at least one. Multi-run bundles list all.
        model_labels: provider/model pairs by stage, e.g.
            {"characterization": "anthropic/claude-...", "diagnosis": "gemini/..."}.
        flags: free-form runtime flags. budget_exhausted is reserved for diagnose.
    """

    model_config = ConfigDict(frozen=True)

    bundle_schema_version: Literal["1.0"] = Field(default=BUNDLE_SCHEMA_VERSION)
    golden_schema_version: str
    pipeline_version: str
    created_at: datetime
    bundle_id: str
    run_ids: list[str] = Field(min_length=1)
    model_labels: dict[str, str] = Field(default_factory=dict)
    flags: dict[str, bool] = Field(default_factory=dict)


def parse_major_minor(version: str) -> tuple[int, int]:
    """Parse 'X.Y' or 'X.Y.Z' into (major, minor). Tolerates suffixes."""
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(f"version {version!r} does not look like 'major.minor'")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"version {version!r} has non-integer components") from e
