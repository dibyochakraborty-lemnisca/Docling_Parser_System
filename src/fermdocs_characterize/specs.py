"""SpecsProvider: per-variable nominal/std_dev/unit lookup.

The schema (golden_schema.yaml) is the single source of truth for vocabulary
and default specs. A dossier may carry per-run overrides under `_specs`.

Layering rules (see DictSpecsProvider.from_schema_with_overrides):
  - Schema provides defaults per variable.
  - Dossier `_specs[var]` is a partial dict; specified fields override schema.
  - Dossier may introduce variables absent from the schema (run-only specs).
  - Setting a dossier field to `null` explicitly clears that field, even if
    the schema had a value.
  - A Spec is emitted only when both nominal and std_dev resolve to non-None
    after merging.

Every Spec carries provenance so downstream agents can reason about how much
to trust a violation: "this nominal came from the batch sheet" is very
different from "this nominal is a schema default."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fermdocs.domain.models import GoldenSchema

SpecProvenance = Literal["schema", "dossier", "merged", "unknown"]


@dataclass(frozen=True)
class Spec:
    nominal: float
    std_dev: float
    unit: str
    provenance: SpecProvenance = "unknown"
    source_ref: str | None = None


class SpecsProvider(Protocol):
    def get(self, variable: str) -> Spec | None: ...


class DictSpecsProvider:
    """Concrete provider backed by a plain dict. Used by fixtures and tests."""

    def __init__(self, specs: dict[str, Spec]) -> None:
        self._specs = dict(specs)

    def get(self, variable: str) -> Spec | None:
        return self._specs.get(variable)

    @classmethod
    def from_dossier(cls, dossier: dict[str, Any]) -> DictSpecsProvider:
        """Build a provider by reading the dossier's `_specs` field (fixture format).

        Real ingestion dossiers won't carry _specs; in production an
        IngestionSpecsProvider will read from a setpoint table.
        """
        raw = dossier.get("_specs", {})
        specs: dict[str, Spec] = {}
        for var, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            if "nominal" not in payload or "std_dev" not in payload:
                continue
            nominal = payload["nominal"]
            std_dev = payload["std_dev"]
            if nominal is None or std_dev is None:
                continue
            specs[var] = Spec(
                nominal=float(nominal),
                std_dev=float(std_dev),
                unit=str(payload.get("unit") or ""),
                provenance="dossier",
                source_ref=payload.get("source_ref"),
            )
        return cls(specs)

    @classmethod
    def from_schema(cls, schema: GoldenSchema) -> DictSpecsProvider:
        """Build a provider from a loaded GoldenSchema.

        Columns lacking nominal or std_dev are skipped.
        """
        specs: dict[str, Spec] = {}
        for col in schema.columns:
            if col.nominal is None or col.std_dev is None:
                continue
            specs[col.name] = Spec(
                nominal=float(col.nominal),
                std_dev=float(col.std_dev),
                unit=col.canonical_unit or "",
                provenance="schema",
            )
        return cls(specs)

    @classmethod
    def from_schema_with_overrides(
        cls, schema: GoldenSchema, dossier: dict[str, Any]
    ) -> DictSpecsProvider:
        """Schema provides defaults; dossier `_specs` overrides field-by-field.

        Merge rules:
          - `dossier._specs[var]` is a partial dict with optional keys
            `nominal`, `std_dev`, `unit`, `source_ref`.
          - Each present key overrides the schema's value for that field.
          - A key set to `null` explicitly clears the schema's value.
          - Variables in the dossier but absent from the schema are emitted
            from the dossier alone (provenance="dossier").
          - A Spec is emitted only when both nominal and std_dev resolve to
            non-None.
          - Provenance: "schema" if no override touched it, "dossier" if the
            schema had no entry at all, "merged" otherwise.
        """
        schema_by_name = schema.by_name()
        overrides_raw = dossier.get("_specs", {}) if isinstance(dossier, dict) else {}
        overrides: dict[str, dict[str, Any]] = {
            k: v for k, v in overrides_raw.items() if isinstance(v, dict)
        }

        specs: dict[str, Spec] = {}
        all_vars = set(schema_by_name) | set(overrides)

        for var in all_vars:
            col = schema_by_name.get(var)
            override = overrides.get(var, {})

            schema_nominal = col.nominal if col else None
            schema_std_dev = col.std_dev if col else None
            schema_unit = (col.canonical_unit if col else None) or ""

            # Field-by-field merge: present in override (incl. None) wins.
            nominal = override["nominal"] if "nominal" in override else schema_nominal
            std_dev = override["std_dev"] if "std_dev" in override else schema_std_dev
            unit_override = override["unit"] if "unit" in override else None
            unit = (unit_override if unit_override is not None else schema_unit) or ""

            if nominal is None or std_dev is None:
                continue

            if col is None:
                provenance: SpecProvenance = "dossier"
            elif override:
                provenance = "merged"
            else:
                provenance = "schema"

            specs[var] = Spec(
                nominal=float(nominal),
                std_dev=float(std_dev),
                unit=unit,
                provenance=provenance,
                source_ref=override.get("source_ref"),
            )
        return cls(specs)
