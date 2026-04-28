from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pint

from fermdocs.domain.models import ConversionStatus

_REGISTRY_FILE = Path(__file__).parent / "registry.txt"


@dataclass
class ConversionResult:
    value_canonical: float | None
    unit_canonical: str | None
    status: ConversionStatus
    error: str | None = None


class UnitConverter:
    def __init__(self, registry: pint.UnitRegistry | None = None) -> None:
        self._ureg = registry or _build_registry()

    def convert(
        self, value: Any, unit_raw: str | None, canonical_unit: str | None
    ) -> ConversionResult:
        if canonical_unit is None:
            return ConversionResult(
                value_canonical=None,
                unit_canonical=None,
                status=ConversionStatus.NOT_APPLICABLE,
            )
        if value is None:
            return ConversionResult(
                value_canonical=None,
                unit_canonical=canonical_unit,
                status=ConversionStatus.FAILED,
                error="value is None",
            )
        try:
            num = float(value)
        except (TypeError, ValueError) as e:
            return ConversionResult(None, canonical_unit, ConversionStatus.FAILED, str(e))
        if unit_raw is None or unit_raw == canonical_unit:
            return ConversionResult(num, canonical_unit, ConversionStatus.OK)
        try:
            q = self._ureg.Quantity(num, unit_raw)
            converted = q.to(canonical_unit)
            return ConversionResult(
                value_canonical=float(converted.magnitude),
                unit_canonical=canonical_unit,
                status=ConversionStatus.OK,
            )
        except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError) as e:
            return ConversionResult(None, canonical_unit, ConversionStatus.FAILED, str(e))


def _build_registry() -> pint.UnitRegistry:
    ureg = pint.UnitRegistry()
    ureg.load_definitions(str(_REGISTRY_FILE))
    return ureg
