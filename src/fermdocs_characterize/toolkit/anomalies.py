"""Generic anomaly detectors run by the characterize layer.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 3.

The 7.5/10 review flagged that sensor / protocol anomalies belong as
metadata, not LLM judgement calls. These detectors are deterministic
and organism-agnostic:

  - detect_instrument_changes: scans run-attributable narratives for
    instrument-name strings; emits when the same measurement_kind has
    different instruments across runs. Generic on any named instrument
    (spectrophotometers, DO probes, pH probes, HPLC, etc.).

  - detect_h0_outliers: per variable, flags runs whose t≈0 value is
    > k·MAD from the cohort median. Works on any time-series.

Step-change detection (the third sub-detector in the plan) is
deferred until we can validate it on multiple bundles — carotenoid
alone risks overfitting the threshold.

These functions are pure: take inputs, return dataclasses. The catalog
wire-up (an ANOMALY_* metric_id with adapter) is a follow-up; for now
the trajectory_analyzer or a standalone runner can call these directly.
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
