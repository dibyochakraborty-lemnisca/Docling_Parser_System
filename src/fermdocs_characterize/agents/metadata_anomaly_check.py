"""Metadata anomaly pre-pass: deterministic Findings before LLM agents see the data.

Reviewer feedback A1 (highest-ROI item from the 7.5/10 review):
  > Hitachi 1900U → LABMAN spectrophotometer change between Batch 3 and
  > Batch 4 is not flagged anywhere... Same issue exists for the unit
  > notation change in Batch 6, the scale change between batches, the
  > Bioreactor A → Bioreactor B switch.

This module is the pipeline-level wire-up of the pure-function detectors
in `fermdocs_characterize.toolkit.anomalies`. It assembles per-run
inputs from a `_BundleView` + the dossier dict + the trajectory list,
and emits structured Findings the synthesizer treats as cross-batch
confounds.

Which detectors fire here today:
  - instrument_changes: from narrative bodies grouped by run_id
  - header_inconsistencies: from golden_columns observations
    (raw_header per (run_id, variable))
  - h0_outliers: from trajectory t≈0 values per (run_id, variable)

Deferred (need per-run dossier fields the manifest doesn't yet carry):
  - scale_changes: requires per-run working_volume_l
  - bioreactor_changes: requires per-run vessel_type / reactor_id

When those fields land in the dossier shape, add the corresponding
blocks below; the toolkit detectors are already in place.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from fermdocs_characterize.agents.catalog_runner import _BundleView
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
    Trajectory,
)
from fermdocs_characterize.toolkit.anomalies import (
    H0Outlier,
    HeaderInconsistency,
    InstrumentChange,
    detect_h0_outliers,
    detect_header_inconsistencies,
    detect_instrument_changes,
)

_log = logging.getLogger(__name__)


def check_metadata_anomalies(
    *,
    char_id: UUID,
    bundle: _BundleView,
    dossier: dict[str, Any],
    narrative_observations: list[dict[str, Any]] | list[Any],
    starting_index: int = 1,
) -> list[Finding]:
    """Run all wired metadata-anomaly detectors and emit Findings.

    Returns a list to APPEND. starting_index controls finding-id namespacing;
    pass `len(existing_findings)` so subsequent IDs sort after the others.
    """
    out: list[Finding] = []
    counter = starting_index

    # --- Instrument changes (from narratives grouped by run) ---
    narratives_by_run = _group_narratives_by_run(narrative_observations)
    if narratives_by_run:
        for change in detect_instrument_changes(narratives_by_run):
            counter += 1
            anchor = _anchor_obs_id(
                bundle, list(change.instruments_by_run.keys())[0]
            )
            if anchor is None:
                continue
            out.append(_instrument_change_finding(
                finding_id=f"{char_id}:F-{counter:04d}",
                change=change,
                anchor_observation_id=anchor,
            ))

    # --- Header inconsistencies (from golden_columns observations) ---
    headers_by_run = _group_headers_by_run(dossier)
    if headers_by_run:
        for hi in detect_header_inconsistencies(headers_by_run):
            counter += 1
            anchor = _anchor_obs_id(
                bundle, list(hi.raw_headers_by_run.keys())[0]
            )
            if anchor is None:
                continue
            out.append(_header_inconsistency_finding(
                finding_id=f"{char_id}:F-{counter:04d}",
                hi=hi,
                anchor_observation_id=anchor,
            ))

    # --- H0 outliers (from trajectories) ---
    h0_by_run = _h0_values_by_run(bundle.trajectories)
    if h0_by_run:
        for outlier in detect_h0_outliers(h0_by_run):
            counter += 1
            anchor = _anchor_obs_id(bundle, outlier.run_id)
            if anchor is None:
                continue
            out.append(_h0_outlier_finding(
                finding_id=f"{char_id}:F-{counter:04d}",
                outlier=outlier,
                anchor_observation_id=anchor,
            ))

    return out


# -----------------------------------------------------------------------------
# Input assembly helpers
# -----------------------------------------------------------------------------


def _group_narratives_by_run(
    narratives: list[Any],
) -> dict[str, list[str]]:
    """Build {run_id: [text, ...]} from narrative observation dicts/models.

    Skips narratives without a run_id (file-level / cross-run).
    """
    by_run: dict[str, list[str]] = defaultdict(list)
    for n in narratives or []:
        if isinstance(n, dict):
            run_id = n.get("run_id")
            text = n.get("text")
        else:
            run_id = getattr(n, "run_id", None)
            text = getattr(n, "text", None)
        if not run_id or not text:
            continue
        by_run[str(run_id)].append(str(text))
    return dict(by_run)


def _group_headers_by_run(
    dossier: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Build {run_id: {canonical_var: raw_header}} from golden_columns.

    Each observation carries `raw_header` (storage layer) +
    `source.locator.run_id`. Multiple observations on the same
    (run, variable) typically share the same raw_header; we take the
    first one we see.
    """
    by_run: dict[str, dict[str, str]] = defaultdict(dict)
    golden = dossier.get("golden_columns") or {}
    for variable, col_data in golden.items():
        if not isinstance(col_data, dict):
            continue
        observations = col_data.get("observations") or []
        for obs in observations:
            raw = obs.get("raw_header") if isinstance(obs, dict) else None
            locator = (obs.get("source") or {}).get("locator") or {} if isinstance(obs, dict) else {}
            run_id = locator.get("run_id")
            if not raw or not run_id:
                continue
            run_key = str(run_id)
            if variable not in by_run[run_key]:
                by_run[run_key][variable] = str(raw)
    return dict(by_run)


def _h0_values_by_run(
    trajectories: list[Trajectory],
) -> dict[str, dict[str, float]]:
    """For each trajectory, take the value at the smallest time as h0."""
    by_run: dict[str, dict[str, float]] = defaultdict(dict)
    for t in trajectories:
        if not t.time_grid or not t.values:
            continue
        # Smallest-time index. time_grid is built sorted in build_trajectories,
        # but defensively pick the min anyway.
        idx = 0
        smallest = t.time_grid[0]
        for i, hr in enumerate(t.time_grid):
            if hr < smallest:
                smallest = hr
                idx = i
        v = t.values[idx]
        if v is None:
            continue
        try:
            by_run[t.run_id][t.variable] = float(v)
        except (TypeError, ValueError):
            continue
    return dict(by_run)


def _anchor_obs_id(bundle: _BundleView, run_id: str) -> str | None:
    """Find any observation_id on this run for the validator's namespace check."""
    for t in bundle.trajectories:
        if t.run_id == run_id and t.source_observation_ids:
            return t.source_observation_ids[0]
    return None


# -----------------------------------------------------------------------------
# Finding constructors
# -----------------------------------------------------------------------------


def _instrument_change_finding(
    *,
    finding_id: str,
    change: InstrumentChange,
    anchor_observation_id: str,
) -> Finding:
    instruments_str = ", ".join(
        f"{r}={i}" for r, i in sorted(change.instruments_by_run.items())
    )
    return Finding(
        finding_id=finding_id,
        type=FindingType.CONTRADICTS,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=(
            f"[METADATA-ANOMALY] {change.instrument_kind} differs across"
            f" runs: {instruments_str}. Cross-batch comparisons on"
            f" measurements from this kind may carry an instrument-bias"
            f" confound."
        ),
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(
            n_observations=len(change.instruments_by_run),
            n_independent_runs=len(change.instruments_by_run),
        ),
        evidence_observation_ids=[anchor_observation_id],
        variables_involved=[],
        run_ids=sorted(change.instruments_by_run.keys()),
        statistics={
            "pattern_kind": "metadata_anomaly",
            "anomaly_kind": "instrument_change",
            "instrument_kind": change.instrument_kind,
            "instruments_by_run": dict(change.instruments_by_run),
            "tier": "A",
        },
    )


def _header_inconsistency_finding(
    *,
    finding_id: str,
    hi: HeaderInconsistency,
    anchor_observation_id: str,
) -> Finding:
    headers_str = ", ".join(
        f"{r}={h!r}" for r, h in sorted(hi.raw_headers_by_run.items())
    )
    return Finding(
        finding_id=finding_id,
        type=FindingType.CONTRADICTS,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=(
            f"[METADATA-ANOMALY] {hi.variable!r} reported under"
            f" different raw headers across runs: {headers_str}. Unit"
            f" notation drift may indicate measurement-protocol change."
        ),
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(
            n_observations=len(hi.raw_headers_by_run),
            n_independent_runs=len(hi.raw_headers_by_run),
        ),
        evidence_observation_ids=[anchor_observation_id],
        variables_involved=[hi.variable],
        run_ids=sorted(hi.raw_headers_by_run.keys()),
        statistics={
            "pattern_kind": "metadata_anomaly",
            "anomaly_kind": "header_inconsistency",
            "variable": hi.variable,
            "raw_headers_by_run": dict(hi.raw_headers_by_run),
            "tier": "A",
        },
    )


def _h0_outlier_finding(
    *,
    finding_id: str,
    outlier: H0Outlier,
    anchor_observation_id: str,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        type=FindingType.COHORT_OUTLIER,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=(
            f"[METADATA-ANOMALY] {outlier.run_id} h≈0 value for"
            f" {outlier.variable!r} is {outlier.run_value:.3g} vs cohort"
            f" median {outlier.cohort_median:.3g} (MAD-score"
            f" {outlier.mad_score:.1f}; threshold 3.0). Initial-condition"
            f" outlier suggests a different inoculum, dilution, or"
            f" measurement protocol on this run."
        ),
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(
            n_observations=1,
            n_independent_runs=1,
        ),
        evidence_observation_ids=[anchor_observation_id],
        variables_involved=[outlier.variable],
        run_ids=[outlier.run_id],
        statistics={
            "pattern_kind": "metadata_anomaly",
            "anomaly_kind": "h0_outlier",
            "variable": outlier.variable,
            "run_value": outlier.run_value,
            "cohort_median": outlier.cohort_median,
            "mad": outlier.mad,
            "mad_score": outlier.mad_score,
            "tier": "A",
        },
    )
