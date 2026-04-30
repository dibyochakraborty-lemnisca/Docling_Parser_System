"""SpecsProvider: per-variable nominal/std_dev/unit lookup.

v1 ships a DictSpecsProvider that reads from the dossier's `_specs` field
(synthetic fixtures) or from a plain dict. Real ingestion will provide an
IngestionSpecsProvider backed by a setpoint table; the protocol is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Spec:
    nominal: float
    std_dev: float
    unit: str


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
            specs[var] = Spec(
                nominal=float(payload["nominal"]),
                std_dev=float(payload["std_dev"]),
                unit=str(payload.get("unit", "")),
            )
        return cls(specs)
