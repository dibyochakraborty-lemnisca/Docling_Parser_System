from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pint

_log = logging.getLogger(__name__)

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
        density_g_per_ml: float | None = None,
    ) -> ConversionResult:
        # De-LABS #0a: a gravimetric/fraction source (%w/w, g/kg) for a VOLUMETRIC
        # concentration target (g/L) is density-dependent. Handle it BEFORE pint and
        # the normalizer — the normalizer's DIMENSIONLESS rule would otherwise
        # silently store the raw %w/w number as g/L (the corruption #0 exists to
        # kill). Refuses legibly when density is absent; never imputes a default.
        grav = self._gravimetric_to_volumetric(
            value, unit_raw, canonical_unit, density_g_per_ml
        )
        if grav is not None:
            return grav
        result = self._convert_with_pint(value, unit_raw, canonical_unit)
        if result.status == ConversionStatus.FAILED and normalizer is not None and unit_raw:
            hint = normalizer.normalize(unit_raw, canonical_unit or "", value)
            result = self.apply_hint(value, unit_raw, canonical_unit, hint)
        return result

    def _gravimetric_to_volumetric(
        self,
        value: Any,
        unit_raw: str | None,
        canonical_unit: str | None,
        density_g_per_ml: float | None,
    ) -> ConversionResult | None:
        """Density-aware concentration conversion, or None when this isn't that case.

        Returns a ConversionResult ONLY when the source is a gravimetric/fraction
        concentration (%w/w, g/kg) and the canonical target is volumetric (g/L):
          - density supplied  -> g/L = value * base * density   (base: %w/w→10, g/kg→1)
          - density absent    -> FAILED with a legible, fix-pointing reason.
        Returns None for every other (unit_raw, canonical) pair so the normal
        pint/normalizer path is unaffected."""
        if not _is_volumetric_conc(canonical_unit):
            return None
        base = _gravimetric_base(unit_raw)
        if base is None:
            return None  # not a gravimetric source -> ordinary conversion path
        if density_g_per_ml is None:
            # Refuse, don't impute. A wrong/default density yields a plausible-but-
            # wrong g/L that passes every range check — the exact silent failure
            # #0 targets. Name density as the missing input so the fix is obvious.
            return ConversionResult(
                value_canonical=None,
                unit_canonical=canonical_unit,
                status=ConversionStatus.FAILED,
                error=(f"{canonical_unit} requires per-run density to convert from "
                       f"'{unit_raw}'; none supplied"),
                via="density_rule",
            )
        try:
            num = float(value)
            dens = float(density_g_per_ml)
        except (TypeError, ValueError) as e:
            return ConversionResult(
                None, canonical_unit, ConversionStatus.FAILED,
                error=f"non-numeric value/density: {e}", via="density_rule",
            )
        return ConversionResult(
            value_canonical=num * base * dens,
            unit_canonical=canonical_unit,
            status=ConversionStatus.OK,
            via="density_rule",
        )

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
            # SCOPED GUARD (de-LABS #0a): "dimensionless, store as-is" is only safe
            # when the canonical TARGET is itself dimensionless (pH, OD, %). For a
            # DIMENSIONED concentration target (g/L) it would silently store a bare/
            # fractional number as g/L — the #0 defect. Refuse instead.
            # IMPORTANT: this is scoped to the *conversion target* being a volumetric
            # concentration. Do NOT widen it to ban dimensionless values in general —
            # density-as-specific-gravity (~1.05, 0b) is legitimately dimensionless
            # and must continue to pass through this branch.
            if _is_volumetric_conc(canonical_unit):
                return ConversionResult(
                    None,
                    canonical_unit,
                    ConversionStatus.FAILED,
                    error=(f"{canonical_unit}: source '{unit_raw}' resolved as "
                           "dimensionless; a concentration target needs a real unit "
                           "(and per-run density for %w/w)"),
                    via=hint.source,
                    hint=hint,
                )
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


def _norm_unit(u: str | None) -> str:
    """Lowercase, strip spaces — so 'g / cm3', '% w/w', 'g/Kg' compare cleanly."""
    return "".join(str(u).split()).lower() if u is not None else ""


_VOLUMETRIC_CONC = {"g/l"}
# Gravimetric/fraction source -> g/L needs density. Base multiplier (then x density):
#   %w/w : g/L = %w/w * density(g/mL) * 10      -> base 10
#   g/kg : g/L = g/kg * density(kg/L=g/mL)      -> base 1
_GRAVIMETRIC_BASE = {
    "%w/w": 10.0, "%": 10.0, "percent": 10.0, "g/100g": 10.0,
    "g/kg": 1.0,
}


def _is_volumetric_conc(canonical_unit: str | None) -> bool:
    return _norm_unit(canonical_unit) in _VOLUMETRIC_CONC


def _gravimetric_base(unit_raw: str | None) -> float | None:
    """Density multiplier base for a gravimetric/fraction source unit, or None."""
    if unit_raw is None:
        return None
    return _GRAVIMETRIC_BASE.get(_norm_unit(unit_raw))


def _build_registry() -> pint.UnitRegistry:
    ureg = pint.UnitRegistry()
    ureg.load_definitions(str(_REGISTRY_FILE))
    return ureg
