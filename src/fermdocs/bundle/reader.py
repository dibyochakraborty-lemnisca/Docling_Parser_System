"""Read-only bundle accessor with version-mismatch enforcement.

The reader refuses to open any directory missing `meta.json` (half-written
bundle). It enforces:
  - bundle_schema_version: exact match with BUNDLE_SCHEMA_VERSION → else
    `BundleSchemaMismatch`
  - golden_schema_version: major mismatch → `GoldenSchemaMajorMismatch`,
    minor mismatch → warning logged, proceed.

Caches the dossier and characterization JSON at __init__ so repeated
fetches don't re-read disk.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from fermdocs.bundle.meta import (
    BUNDLE_SCHEMA_VERSION,
    BundleMeta,
    BundleSchemaMismatch,
    GoldenSchemaMajorMismatch,
    parse_major_minor,
)

logger = logging.getLogger(__name__)


class BundleNotReady(Exception):
    """Bundle directory exists but has no meta.json (half-written or corrupt)."""


class BundleReader:
    """Read-only accessor for a bundle directory.

    Eager-loads dossier and characterization at construction so all subsequent
    fetches are in-memory.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        current_golden_schema_version: str | None = None,
    ) -> None:
        self._dir = Path(bundle_dir)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"bundle dir not found: {self._dir}")
        meta_path = self._dir / "meta.json"
        if not meta_path.exists():
            raise BundleNotReady(
                f"meta.json missing in {self._dir}; bundle may be half-written"
            )
        # Pre-check bundle_schema_version BEFORE Pydantic validation so a
        # mismatched bundle gets a clear domain error instead of a generic
        # Literal-rejection ValidationError.
        raw = json.loads(meta_path.read_text())
        bundle_v = raw.get("bundle_schema_version")
        if bundle_v != BUNDLE_SCHEMA_VERSION:
            raise BundleSchemaMismatch(
                f"bundle_schema_version mismatch: "
                f"bundle={bundle_v!r}, reader={BUNDLE_SCHEMA_VERSION!r}"
            )
        self._meta = BundleMeta.model_validate(raw)
        self._enforce_versions(current_golden_schema_version)

        # Eager loads. Subsequent fetches are pure in-memory dict accesses.
        dossier_path = self._dir / "dossier.json"
        self._dossier: dict[str, Any] | None = (
            json.loads(dossier_path.read_text()) if dossier_path.exists() else None
        )
        char_path = self._dir / "characterization" / "characterization.json"
        self._characterization_text: str | None = (
            char_path.read_text() if char_path.exists() else None
        )

    def _enforce_versions(self, current_golden: str | None) -> None:
        if self._meta.bundle_schema_version != BUNDLE_SCHEMA_VERSION:
            raise BundleSchemaMismatch(
                f"bundle_schema_version mismatch: "
                f"bundle={self._meta.bundle_schema_version!r}, "
                f"reader={BUNDLE_SCHEMA_VERSION!r}"
            )
        if current_golden is None:
            return
        try:
            bundle_mm = parse_major_minor(self._meta.golden_schema_version)
            current_mm = parse_major_minor(current_golden)
        except ValueError as e:
            logger.warning("could not parse golden_schema_version: %s", e)
            return
        if bundle_mm[0] != current_mm[0]:
            raise GoldenSchemaMajorMismatch(
                f"golden_schema major mismatch: "
                f"bundle={self._meta.golden_schema_version!r}, "
                f"current={current_golden!r}"
            )
        if bundle_mm[1] != current_mm[1]:
            warnings.warn(
                f"golden_schema minor mismatch (bundle={self._meta.golden_schema_version!r}, "
                f"current={current_golden!r}); spec references may be stale",
                stacklevel=2,
            )

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def meta(self) -> BundleMeta:
        return self._meta

    def get_dossier(self) -> dict[str, Any]:
        if self._dossier is None:
            raise FileNotFoundError(f"no dossier.json in {self._dir}")
        return self._dossier

    def get_characterization_json(self) -> str:
        """Raw JSON text of CharacterizationOutput.

        Stage 1 returns the JSON string and lets callers validate against
        their own pydantic model. Avoids cross-package coupling.
        """
        if self._characterization_text is None:
            raise FileNotFoundError(
                f"no characterization/characterization.json in {self._dir}"
            )
        return self._characterization_text

    def has_diagnosis(self) -> bool:
        return (self._dir / "diagnosis" / "diagnosis.json").exists()

    def get_diagnosis_json(self) -> str:
        path = self._dir / "diagnosis" / "diagnosis.json"
        if not path.exists():
            raise FileNotFoundError(f"no diagnosis/diagnosis.json in {self._dir}")
        return path.read_text()
