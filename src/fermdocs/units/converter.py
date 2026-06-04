from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pint

_log = logging.getLogger(__name__)

# A converted value this many times the variable's nominal is treated as a
# suspected unit mislabel (e.g. a column tagged g/L whose values are really
# mg/L, inflating by 1000x). Override via FERMDOCS_UNIT_PLAUSIBILITY_FACTOR.
_DEFAULT_PLAUSIBILITY_FACTOR = 50.0

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
        nominal: float | None = None,
    ) -> ConversionResult:
        result = self._convert_with_pint(value, unit_raw, canonical_unit)
        if result.status == ConversionStatus.FAILED and normalizer is not None and unit_raw:
            hint = normalizer.normalize(unit_raw, canonical_unit or "", value)
            result = self.apply_hint(value, unit_raw, canonical_unit, hint)
        return self._correct_mislabeled_unit(result, value, unit_raw, canonical_unit, nominal)

    def _correct_mislabeled_unit(
        self,
        result: ConversionResult,
        value: Any,
        unit_raw: str | None,
        canonical_unit: str | None,
        nominal: float | None,
    ) -> ConversionResult:
        """Undo a unit conversion that produced a physically implausible value.

        When a source column's declared unit is wrong (e.g. IndPenSim's PAA/NH3
        offline columns tagged 'g/L' but holding mg/L values), pint faithfully
        scales by the conversion factor and emits an absurd magnitude that
        characterization then flags as a fake anomaly. If the variable has a
        nominal and the converted value is >FACTOR x nominal while the
        UN-converted value is plausible, the label was almost certainly wrong by
        exactly that factor: treat the source value as already canonical.
        """
        if (
            result.status != ConversionStatus.OK
            or result.value_canonical is None
            or not nominal
            or unit_raw is None
            or canonical_unit is None
            or unit_raw == canonical_unit
        ):
            return result
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return result
        try:
            factor = float(os.environ.get("FERMDOCS_UNIT_PLAUSIBILITY_FACTOR", "") or _DEFAULT_PLAUSIBILITY_FACTOR)
        except ValueError:
            factor = _DEFAULT_PLAUSIBILITY_FACTOR
        ceiling = abs(nominal) * factor
        converted = abs(result.value_canonical)
        if converted > ceiling and abs(raw) <= ceiling:
            _log.warning(
                "unit-normalizer: %r->%r yielded %.4g (>%.0fx nominal %.4g) but the source "
                "value %.4g is plausible; treating it as already %s (suspected mislabeled unit)",
                unit_raw, canonical_unit, result.value_canonical, factor, nominal, raw, canonical_unit,
            )
            return replace(result, value_canonical=raw, via="unit_corrected")
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
