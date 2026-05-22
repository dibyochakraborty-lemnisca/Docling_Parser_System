"""Generic metadata-anomaly detectors run by the characterize layer.

Plan refs:
  - plans/2026-05-07-rigour-and-actionability.md commit 3 (initial:
    instrument-change, h0-outlier).
  - 7.5/10 reviewer feedback A1 (this commit): scale-change,
    bioreactor-change, header-inconsistency.

The reviewer's framing: metadata anomalies are deterministic facts the
system should detect *before any LLM sees the data*. A wet-lab scientist
will catch Hitachi→LABMAN spectrophotometer swap or 2.5 L→3.5 L→1 L
working-volume drift in 30 seconds; the system has to flag them at
pre-pass time so cross-batch comparisons surface the confound.

Detectors:

  - detect_instrument_changes: scans run-attributable narratives for
    instrument-name strings; emits when the same measurement_kind has
    different instruments across runs.

  - detect_h0_outliers: per variable, flags runs whose t≈0 value is
    > k·MAD from the cohort median.

  - detect_scale_changes: emits when working_volume_l differs across
    runs by more than `relative_threshold` (default 0.10 = 10%).
    Volumetric metrics (Qp, OUR per-volume) are not directly comparable
    across scale changes — the flag forces explicit acknowledgement.

  - detect_bioreactor_changes: emits when vessel_type / reactor_id
    differs across runs. Mechanistically distinct from instrument
    change — same OD probe on a different reactor still has different
    geometry, kLa, mixing time.

  - detect_header_inconsistencies: emits when the same canonical
    variable is reported under different raw column headers across
    runs (e.g. 'WCW (mg/3 mL)' on 5 runs, 'Wet cell weight (mg)' on 1
    — the second batch dropped the dilution annotation).

All detectors are pure: take inputs, return dataclasses. The catalog
wire-up (an ANOMALY_* metric_id with adapter) is a follow-up; for now
the trajectory_analyzer or a standalone runner can call these directly.

Step-change detection is deliberately omitted — carotenoid alone
risks overfitting the threshold; defer until multi-bundle validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# -----------------------------------------------------------------------------
# Instrument-change detector
# -----------------------------------------------------------------------------

# Common instrument keywords that show up in run narratives. We match
# capitalised tokens that follow these keywords — a brand name or model
# string sitting next to the keyword. The pattern is intentionally
# narrow: false positives here become user-facing findings, so we
# prefer recall < precision on the first pass. Add keywords as bundles
# expose new instrument vocabulary.
INSTRUMENT_KEYWORDS = (
    "spectrophotometer",
    "spectrometer",
    "hplc",
    "gc-ms",
    "gcms",
    "do probe",
    "oxygen probe",
    "ph probe",
    "ph electrode",
    "thermocouple",
    "flowmeter",
    "mass spectrometer",
)

# Brand-shaped tokens: either Mixed-case ("Hitachi", "Mettler-Toledo")
# or all-caps with ≥4 chars ("LABMAN", "HAMILTON"). Three-letter
# all-caps tokens ("WCW", "DCW", "OD") are domain abbreviations, not
# brands — excluded by the ≥4-char floor on all-caps. Mixed-case
# requires ≥1 lowercase letter to filter out "WCW600" etc.
_BRAND_RE = re.compile(
    r"\b("
    r"[A-Z][a-z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]+)?"  # Mixed-case
    r"|"
    r"[A-Z]{4,}(?:\s+[A-Z][A-Za-z0-9\-]+)?"                    # ALL-CAPS≥4
    r")\b"
)


@dataclass(frozen=True)
class InstrumentChange:
    """One named instrument-kind that varies across runs.

    `instruments_by_run` maps run_id → the instrument string detected.
    Surfaced as a finding with summary like 'spectrophotometer differs
    across runs: RUN-1=Hitachi, RUN-2=LABMAN'.
    """

    instrument_kind: str
    instruments_by_run: dict[str, str]


def detect_instrument_changes(
    narratives_by_run: dict[str, list[str]],
) -> list[InstrumentChange]:
    """Find instrument-kinds whose named instrument differs across runs.

    `narratives_by_run`: per-run free-text narrative bodies. We scan
    each kind keyword and harvest the nearest brand-shaped token; if
    >1 distinct brand appears across runs for the same kind, emit.

    Returns empty list when no changes detected (clean penicillin path).
    """
    if len(narratives_by_run) < 2:
        return []

    out: list[InstrumentChange] = []
    for kind in INSTRUMENT_KEYWORDS:
        per_run: dict[str, str] = {}
        for run_id, texts in narratives_by_run.items():
            for text in texts:
                low = text.lower()
                idx = low.find(kind)
                if idx < 0:
                    continue
                # Look in a window before+after the keyword for a brand-shaped token.
                window_start = max(0, idx - 60)
                window_end = min(len(text), idx + len(kind) + 60)
                window = text[window_start:window_end]
                brands = _BRAND_RE.findall(window)
                # Drop the keyword itself if it happens to capture (e.g. 'DO').
                brands = [b for b in brands if b.lower() not in kind]
                if brands:
                    per_run[run_id] = brands[0]
                    break
        unique_brands = set(per_run.values())
        if len(unique_brands) >= 2:
            out.append(
                InstrumentChange(
                    instrument_kind=kind,
                    instruments_by_run=dict(per_run),
                )
            )
    return out


# -----------------------------------------------------------------------------
# H0 outlier detector
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class H0Outlier:
    """A run whose t≈0 value for a variable is far from cohort median.

    `mad_score`: |run_value - cohort_median| / MAD. >3 is the standard
    robust-stats outlier threshold.
    """

    variable: str
    run_id: str
    run_value: float
    cohort_median: float
    mad: float
    mad_score: float


def _median_absolute_deviation(values: np.ndarray) -> float:
    """MAD = median(|x - median(x)|). Returns 0 for fully constant input."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


# -----------------------------------------------------------------------------
# Scale-change detector
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleChange:
    """Working-volume varies across runs by more than relative threshold.

    `volumes_by_run`: run_id → working_volume_l. Sorted by run_id when
    rendered. `min_l` / `max_l` are convenience for the summary string.
    """

    volumes_by_run: dict[str, float]
    min_l: float
    max_l: float
    relative_spread: float  # (max - min) / min


def detect_scale_changes(
    volumes_by_run: dict[str, float | None],
    *,
    relative_threshold: float = 0.10,
) -> list[ScaleChange]:
    """Emit when working volumes vary by more than relative_threshold.

    `volumes_by_run`: per-run working volumes in litres. None values
    are dropped (run that didn't declare a volume). With <2 valid runs
    we can't compare — returns empty.

    Default 10% threshold: 2.5 L → 2.75 L (10%) is borderline; 2.5 L →
    3.5 L (40%) is a clear scale change. Tighten via the parameter when
    a study is supposed to be at fixed scale.
    """
    valid = {
        rid: float(v)
        for rid, v in volumes_by_run.items()
        if v is not None and np.isfinite(v) and v > 0
    }
    if len(valid) < 2:
        return []
    vmin = min(valid.values())
    vmax = max(valid.values())
    spread = (vmax - vmin) / vmin
    if spread < relative_threshold:
        return []
    return [
        ScaleChange(
            volumes_by_run=dict(valid),
            min_l=vmin,
            max_l=vmax,
            relative_spread=spread,
        )
    ]


# -----------------------------------------------------------------------------
# Bioreactor-change detector
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BioreactorChange:
    """Vessel / reactor identity differs across runs.

    `reactors_by_run`: run_id → reactor identifier (e.g. 'BIOREACTOR_A').
    Different reactors mean different geometry; volumetric and
    mass-transfer metrics aren't directly comparable.
    """

    reactors_by_run: dict[str, str]


def detect_bioreactor_changes(
    reactors_by_run: dict[str, str | None],
) -> list[BioreactorChange]:
    """Emit when ≥2 distinct reactor ids appear across runs.

    `reactors_by_run`: per-run reactor identifier (vessel_type,
    equipment_id, or whatever the dossier exposes). None / empty are
    dropped. Trims whitespace and lowercases for comparison so 'BIOREACTOR_A'
    and 'Bioreactor A' aren't flagged as different.
    """
    cleaned: dict[str, str] = {}
    for rid, name in reactors_by_run.items():
        if not name:
            continue
        s = str(name).strip()
        if not s:
            continue
        cleaned[rid] = s
    if len(cleaned) < 2:
        return []
    # Normalise underscores/spaces and case for the equality check so
    # 'BIOREACTOR_A' and 'Bioreactor A' don't fire as different vessels.
    distinct = {
        re.sub(r"[\s_]+", " ", v).strip().lower() for v in cleaned.values()
    }
    if len(distinct) < 2:
        return []
    return [BioreactorChange(reactors_by_run=cleaned)]


# -----------------------------------------------------------------------------
# Header-inconsistency detector
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderInconsistency:
    """Same canonical variable, different raw headers across runs.

    `variable`: canonical golden-column name (e.g. 'wcw_g_l').
    `raw_headers_by_run`: run_id → the raw header string the parser saw.
    Surfaces unit-notation drift like 'WCW (mg/3 mL)' vs 'Wet cell weight (mg)'.
    """

    variable: str
    raw_headers_by_run: dict[str, str]


def detect_header_inconsistencies(
    raw_headers_by_run_and_variable: dict[str, dict[str, str]],
) -> list[HeaderInconsistency]:
    """Per canonical variable, emit when raw headers differ across runs.

    `raw_headers_by_run_and_variable`: {run_id: {canonical_var: raw_header}}.
    Caller assembles this from the mapping layer (raw → golden column
    bookkeeping). Drops runs where the variable isn't present.

    Whitespace and case differences alone don't fire (likely OCR noise);
    only structurally distinct strings count.
    """
    if not raw_headers_by_run_and_variable:
        return []

    # Reshape: variable → [(run_id, raw_header)]
    by_var: dict[str, list[tuple[str, str]]] = {}
    for run_id, var_map in raw_headers_by_run_and_variable.items():
        for var, raw in var_map.items():
            if not raw or not str(raw).strip():
                continue
            by_var.setdefault(var, []).append((run_id, str(raw).strip()))

    out: list[HeaderInconsistency] = []
    for var, pairs in by_var.items():
        if len(pairs) < 2:
            continue
        # Normalise casing + whitespace for the equality check; preserve
        # original strings in the output.
        normed = {re.sub(r"\s+", " ", raw).strip().lower() for _, raw in pairs}
        if len(normed) < 2:
            continue
        out.append(
            HeaderInconsistency(
                variable=var,
                raw_headers_by_run=dict(pairs),
            )
        )
    return out


# -----------------------------------------------------------------------------
# H0 outlier detector
# -----------------------------------------------------------------------------


def detect_h0_outliers(
    h0_values_by_run: dict[str, dict[str, float]],
    *,
    mad_threshold: float = 3.0,
) -> list[H0Outlier]:
    """Per-variable: flag runs > mad_threshold MADs from cohort median at t≈0.

    `h0_values_by_run`: {run_id: {variable: value_at_t0}}. Caller's
    responsibility to pick the t≈0 sample (smallest t in trajectory,
    or interpolated to t=0).

    Skips variables with <3 runs or zero MAD (constant cohort — any
    deviation would be infinite). Returns empty when no outliers found.
    """
    if not h0_values_by_run:
        return []

    # Reshape: variable → [(run_id, value)]
    by_var: dict[str, list[tuple[str, float]]] = {}
    for run_id, var_map in h0_values_by_run.items():
        for var, val in var_map.items():
            if val is None or not np.isfinite(val):
                continue
            by_var.setdefault(var, []).append((run_id, float(val)))

    out: list[H0Outlier] = []
    for var, pairs in by_var.items():
        if len(pairs) < 3:
            continue
        values = np.asarray([v for _, v in pairs], dtype=float)
        med = float(np.median(values))
        mad = _median_absolute_deviation(values)
        if mad == 0.0:
            continue
        for run_id, val in pairs:
            score = abs(val - med) / mad
            if score > mad_threshold:
                out.append(
                    H0Outlier(
                        variable=var,
                        run_id=run_id,
                        run_value=val,
                        cohort_median=med,
                        mad=mad,
                        mad_score=score,
                    )
                )
    return out
