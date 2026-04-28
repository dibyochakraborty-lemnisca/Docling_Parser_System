from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pint

from fermdocs.domain.models import ConversionStatus
from fermdocs.units.normalizer import (
    NormalizationAction,
    NormalizationHint,
    UnitNormalizer,
)

_REGISTRY_FILE = Path(__file__).parent / "registry.txt"


@dataclass
class ConversionResult:
    value_canonical: float | None
    unit_canonical: str | None
    status: ConversionStatus
    error: str | None = None
    via: str = "pint"  # 'pint' | 'rule_based' | 'llm' | 'chain' | 'not_applicable'
    hint: NormalizationHint | None = None


class UnitConverter:
    def __init__(self, registry: pint.UnitRegistry | None = None) -> None:
        self._ureg = registry or _build_registry()

    def convert(
        self,
        value: Any,
        unit_raw: str | None,
        canonical_unit: str | None,
        normalizer: UnitNormalizer | None = None,
    ) -> ConversionResult:
        result = self._convert_with_pint(value, unit_raw, canonical_unit)
        if result.status == ConversionStatus.FAILED and normalizer is not None and unit_raw:
            hint = normalizer.normalize(unit_raw, canonical_unit or "", value)
            return self.apply_hint(value, unit_raw, canonical_unit, hint)
        return result

    def apply_hint(
        self,
        value: Any,
        unit_raw: str | None,
        canonical_unit: str | None,
        hint: NormalizationHint,
    ) -> ConversionResult:
        if hint.action == NormalizationAction.USE_PINT_EXPR:
            if not hint.pint_expr:
                return ConversionResult(
                    None,
                    canonical_unit,
                    ConversionStatus.FAILED,
                    error="hint missing pint_expr",
                    via=hint.source,
                    hint=hint,
                )
            retried = self._convert_with_pint(value, hint.pint_expr, canonical_unit)
            return replace(
                retried,
                via=hint.source if retried.status == ConversionStatus.OK else retried.via,
                hint=hint,
            )
        if hint.action == NormalizationAction.DIMENSIONLESS:
            try:
                num = float(value)
            except (TypeError, ValueError) as e:
                return ConversionResult(
                    None,
                    canonical_unit,
                    ConversionStatus.FAILED,
                    error=str(e),
                    via=hint.source,
                    hint=hint,
                )
            return ConversionResult(
                value_canonical=num,
                unit_canonical=canonical_unit,
                status=ConversionStatus.OK,
                via=hint.source,
                hint=hint,
            )
        # UNCONVERTIBLE
        return ConversionResult(
            value_canonical=None,
            unit_canonical=canonical_unit,
            status=ConversionStatus.FAILED,
            error=f"unconvertible: {hint.rationale}",
            via=hint.source,
            hint=hint,
        )

    def _convert_with_pint(
        self, value: Any, unit_raw: str | None, canonical_unit: str | None
    ) -> ConversionResult:
        if canonical_unit is None:
            return ConversionResult(
                value_canonical=None,
                unit_canonical=None,
                status=ConversionStatus.NOT_APPLICABLE,
                via="not_applicable",
            )
        if value is None:
            return ConversionResult(
                value_canonical=None,
                unit_canonical=canonical_unit,
                status=ConversionStatus.FAILED,
                error="value is None",
                via="pint",
            )
        try:
            num = float(value)
        except (TypeError, ValueError) as e:
            return ConversionResult(
                None, canonical_unit, ConversionStatus.FAILED, str(e), via="pint"
            )
        if unit_raw is None or unit_raw == canonical_unit:
            return ConversionResult(
                num, canonical_unit, ConversionStatus.OK, via="pint"
            )
        try:
            q = self._ureg.Quantity(num, unit_raw)
            converted = q.to(canonical_unit)
            return ConversionResult(
                value_canonical=float(converted.magnitude),
                unit_canonical=canonical_unit,
                status=ConversionStatus.OK,
                via="pint",
            )
        except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError) as e:
            return ConversionResult(
                None, canonical_unit, ConversionStatus.FAILED, str(e), via="pint"
            )


def _build_registry() -> pint.UnitRegistry:
    ureg = pint.UnitRegistry()
    ureg.load_definitions(str(_REGISTRY_FILE))
    return ureg
