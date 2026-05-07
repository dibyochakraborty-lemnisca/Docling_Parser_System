"""Deterministic per-run catalog runner.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 1.

Replaces the trajectory-analyzer LLM as the iteration boundary for
catalog metrics. Before this module: the LLM was told 'emit one
pattern per applicable run' and silently iterated wrong on multi-run
bundles. After: the runner enumerates every (metric_id × run_id) pair
that applies, calls the toolkit_fn, and emits one Finding per pair —
deterministically, without an LLM.

Flow:

     bundle.run_ids                 ready_entries()
           │                              │
           └──────┐         ┌─────────────┘
                  ▼         ▼
         applicable_metric_run_pairs(bundle)
                  │
                  ▼
            for (metric, run):
                  │
                  ▼
          adapter = ADAPTERS[metric.metric_id]
                  │
                  ▼                       ┌─ raises  → data_gap (tool error)
          adapter(bundle, run_id)  ──────┤
                  │                       └─ returns None → data_gap (input gap)
                  ▼
              statistics  → emit Finding(computed_metric)

**A2 fix — pre-flight import check:** `MetricCatalogRunner.__init__`
imports every `toolkit_fn` once. A dep regression that today produces
bundle-wide "computation failed due to tool error" data_gaps now
aborts characterize loud with the failing module name.

**Q1 fix — shared iteration helper:** `applicable_metric_run_pairs` is
used by this runner AND by `symmetry_check.py` (commit 5) so the two
sides of the symmetry contract iterate identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator

from fermdocs_characterize.agents.metric_catalog import (
    CatalogEntry,
    KINETICS_METRICS,
    MASS_TRANSFER_METRICS,
    METABOLIC_METRICS,
    ready_entries,
)
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
    Trajectory,
)

_log = logging.getLogger(__name__)


# Cross-run metrics emit ONE finding spanning all applicable runs;
# everything else emits one per (metric × run). Today's catalog has only
# A19/A20/A21 in this category. Symmetry check skips these.
CROSS_RUN_METRIC_IDS: frozenset[str] = frozenset({"A19", "A20", "A21"})


@dataclass
class _BundleView:
    """Tight wrapper around the inputs the catalog runner needs.

    Decoupled from CharacterizationOutput so tests can construct it
    cheaply, and so a future migration to a different bundle shape
    doesn't require updating every adapter.
    """

    characterization_id: str
    run_ids: list[str]
    trajectories: list[Trajectory]
    organism: str | None
    process_family: str | None

    def trajectory(self, run_id: str, variable: str) -> Trajectory | None:
        """Return the single Trajectory for (run_id, variable) or None."""
        for t in self.trajectories:
            if t.run_id == run_id and t.variable == variable:
                return t
        return None

    def variables_for(self, run_id: str) -> set[str]:
        return {t.variable for t in self.trajectories if t.run_id == run_id}


# Adapter signature: (bundle, run_id) -> statistics dict | None.
# - Returning a dict means "computed successfully, here are the stats."
# - Returning None means "applicability precondition failed, emit data_gap
#   with a reason rather than a tool-error data_gap."
# - Raising means "unexpected toolkit error; runner converts to data_gap
#   with the exception message."
AdapterFn = Callable[["_BundleView", str], dict | None]

# Adapters are registered lazily so this module doesn't pull in pandas
# at import time (the pre-flight check below imports them deliberately).
_ADAPTERS: dict[str, AdapterFn] = {}


def _register_adapter(metric_id: str):
    def deco(fn: AdapterFn) -> AdapterFn:
        if metric_id in _ADAPTERS:
            raise ValueError(f"duplicate adapter for {metric_id}")
        _ADAPTERS[metric_id] = fn
        return fn
    return deco


def applicable_metric_run_pairs(
    bundle: _BundleView,
) -> Iterator[tuple[str, str | None]]:
    """Enumerate every (metric_id, run_id) pair the runner SHOULD attempt.

    For per-run metrics: yields (metric_id, run_id) for each run.
    For cross-run metrics: yields (metric_id, None) once per metric.

    Used by both `MetricCatalogRunner.compute_all` (commit 1) and
    `check_symmetry` (commit 5) so the iteration is single-source.

    Does NOT check applicability (whether inputs are present); that's the
    adapter's job. This helper just enumerates the universe; symmetry
    check downstream compares emitted findings against this universe to
    detect tool gaps.
    """
    for entry in ready_entries():
        if entry.metric_id in CROSS_RUN_METRIC_IDS:
            yield (entry.metric_id, None)
            continue
        for run_id in bundle.run_ids:
            yield (entry.metric_id, run_id)


class MetricCatalogRunner:
    """Deterministic, side-effect-free runner over the metric catalog.

    Construction does an A2 pre-flight: imports every ready entry's
    `toolkit_fn` once. If any import fails (missing dep, syntax error,
    renamed function), raises RuntimeError with the failing module path.
    This catches the same regression class that today produces a bundle
    full of "computation failed due to tool error" data_gaps. Loud
    failure is correct here — silent degradation hides real bugs.

    `compute_all(bundle)` returns one Finding per applicable
    (metric × run) pair: either `computed_metric` with statistics, or
    `data_gap` with a reason. Idempotent: same input → same output
    (Q2 invariant; pinned by tests).
    """

    def __init__(self) -> None:
        # A2 pre-flight: try to import every ready toolkit_fn now.
        # First import failure aborts construction with the failing
        # module + entry, so characterize never silently degrades.
        # We also force adapter import so registration happens before
        # compute_all runs.
        from fermdocs_characterize.agents import (  # noqa: F401
            catalog_runner_adapters,  # registers adapters as side effect
        )
        for entry in ready_entries():
            try:
                entry.resolve_toolkit_fn()
            except Exception as exc:
                raise RuntimeError(
                    f"catalog runner pre-flight import failed for"
                    f" {entry.metric_id} (toolkit_fn={entry.toolkit_fn!r}):"
                    f" {type(exc).__name__}: {exc}."
                    f" This would have produced a silent 'computation failed"
                    f" due to tool error' data_gap; aborting characterize"
                    f" loud instead. Fix the toolkit module or mark the"
                    f" catalog entry status='pending'."
                ) from exc
            if entry.metric_id not in _ADAPTERS:
                # Soft-skip: ready entry with no adapter falls through to
                # the LLM trajectory analyzer (existing path). Logged at
                # WARNING so a missing adapter is visible but not fatal.
                # Once an adapter lands, the deterministic runner takes
                # over for this metric_id; the analyzer's prompt rewrite
                # (commit 1 below) ensures it doesn't re-emit.
                _log.warning(
                    "catalog runner: no adapter for ready entry %s;"
                    " falling through to LLM trajectory analyzer.",
                    entry.metric_id,
                )
                continue

    def compute_all(self, bundle: _BundleView) -> list[Finding]:
        """Iterate every (metric, run) pair; emit one Finding per pair.

        Cross-run metrics emit ONE finding spanning all runs.
        Adapter return value drives the Finding.type:
          - dict       → computed_metric, statistics from the dict
          - None       → data_gap with reason 'precondition not met'
          - exception  → data_gap with reason 'tool error: <message>'

        Every emitted Finding carries at least one real observation_id
        from the bundle's trajectories (the validator rejects findings
        that cite IDs not in the bundle registry). When the bundle has
        no observations on the relevant run we DROP the finding rather
        than emit something that will fail downstream validation.
        Logged at WARNING for the dev to see why a metric silently
        didn't surface.

        finding_id is deterministic given input order: this lets two
        runs of the same characterize stage compare equal.
        """
        findings: list[Finding] = []
        finding_counter = 0

        for metric_id, run_id in applicable_metric_run_pairs(bundle):
            # Soft-skip metrics with no adapter (pre-flight already warned).
            # Symmetry check (commit 5) will not fire on these because
            # neither this runner nor the LLM produces deterministic
            # findings for them.
            if metric_id not in _ADAPTERS:
                continue
            finding_counter += 1
            adapter = _ADAPTERS[metric_id]
            finding_id = (
                f"{bundle.characterization_id}:F-CR-{finding_counter:04d}"
            )
            entry = next(e for e in ready_entries() if e.metric_id == metric_id)

            try:
                stats = adapter(bundle, run_id) if run_id is not None else adapter(bundle, None)
            except Exception as exc:
                _log.debug(
                    "catalog runner: %s on run=%s raised %s: %s",
                    metric_id, run_id, type(exc).__name__, exc,
                )
                gap = self._data_gap(
                    finding_id=finding_id,
                    metric_id=metric_id,
                    run_id=run_id,
                    reason=f"tool error: {type(exc).__name__}: {exc}",
                    tier=entry.tier,
                    bundle=bundle,
                )
                if gap is not None:
                    findings.append(gap)
                continue

            if stats is None:
                gap = self._data_gap(
                    finding_id=finding_id,
                    metric_id=metric_id,
                    run_id=run_id,
                    reason="precondition not met (missing variable or insufficient points)",
                    tier=entry.tier,
                    bundle=bundle,
                )
                if gap is not None:
                    findings.append(gap)
                continue

            # A3 fix: adapter signaled a config mismatch (process-family
            # YAML routes to a variable not present in the bundle).
            # Convert to a [CONFIG_MISMATCH] data_gap so the user sees a
            # diagnostic message naming the YAML key and the available
            # variables, not a generic 'precondition not met'.
            if stats.get("_config_mismatch"):
                gap = self._data_gap(
                    finding_id=finding_id,
                    metric_id=metric_id,
                    run_id=run_id,
                    reason=stats.get("_config_mismatch_reason")
                    or "process-family config mismatch",
                    tier=entry.tier,
                    pattern_kind="config_mismatch",
                    bundle=bundle,
                )
                if gap is not None:
                    findings.append(gap)
                continue

            run_ids = [run_id] if run_id is not None else list(bundle.run_ids)
            variables = list(stats.pop("_variables_used", [])) or _infer_variables(
                entry, bundle, run_id
            )
            severity = _severity_for(metric_id, stats)
            obs_ids = _collect_observation_ids(bundle, run_ids, variables)
            if not obs_ids:
                # No real observation_ids to cite — the validator would
                # reject this finding. Drop rather than emit something
                # that fails downstream. WARNING so the dev sees why a
                # computed metric didn't surface.
                _log.warning(
                    "catalog runner: %s on %s computed but no"
                    " observation_ids resolved (variables=%s); dropping"
                    " finding to avoid validator failure.",
                    metric_id, run_ids, variables,
                )
                continue
            findings.append(Finding(
                finding_id=finding_id,
                type=FindingType.KINETIC_ANOMALY,  # generic; metric_id is the discriminator
                severity=severity,
                tier=Tier(entry.tier),
                summary=_summary_for(metric_id, run_id, stats),
                confidence=0.85,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(
                    n_observations=int(stats.get("n_observations", 0)),
                    n_independent_runs=len(run_ids),
                ),
                evidence_observation_ids=obs_ids,
                variables_involved=variables,
                run_ids=run_ids,
                statistics={
                    **stats,
                    "pattern_kind": "computed_metric",
                    "metric_id": metric_id,
                    "tier": entry.tier,
                },
            ))

        return findings

    @staticmethod
    def _data_gap(
        *,
        finding_id: str,
        metric_id: str,
        run_id: str | None,
        reason: str,
        tier: str,
        pattern_kind: str = "data_gap",
        bundle: _BundleView | None = None,
    ) -> Finding | None:
        """Emit a non-computed Finding. `pattern_kind` discriminates:
          - 'data_gap': inputs missing or precondition not met
          - 'config_mismatch': process-family YAML routes to a variable
            not in the bundle (A3 fix).
          - 'symmetry_violation': asymmetric coverage (commit 5).
        Severity stays INFO; the synthesizer should treat all of these
        as non-evidence, just diagnostic notes.

        Every Finding requires ≥1 evidence_observation_id that resolves
        through the bundle's registry (validator). For a data_gap we
        don't have a 'real' observation tied to a successful computation;
        we cite ANY observation from a trajectory on the same run as a
        diagnostic anchor. If the bundle has zero trajectories on this
        run, returns None and the caller drops the finding entirely.
        """
        run_label = run_id if run_id is not None else "cross-run"
        prefix = (
            "[CONFIG_MISMATCH] " if pattern_kind == "config_mismatch" else ""
        )

        # Anchor to a real observation_id so the validator passes. Pull
        # from any trajectory on this run; cross-run gaps pull from any
        # trajectory in the bundle.
        obs_ids: list[str] = []
        if bundle is not None:
            if run_id is not None:
                for t in bundle.trajectories:
                    if t.run_id == run_id and t.source_observation_ids:
                        obs_ids = [t.source_observation_ids[0]]
                        break
            else:
                for t in bundle.trajectories:
                    if t.source_observation_ids:
                        obs_ids = [t.source_observation_ids[0]]
                        break
        if not obs_ids:
            # No anchor available — dropping is correct (alternative
            # would be to emit a finding the validator rejects).
            _log.warning(
                "catalog runner: data_gap for %s on %s has no"
                " observation_id anchor; dropping (bundle has no"
                " trajectories on this run).",
                metric_id, run_label,
            )
            return None

        return Finding(
            finding_id=finding_id,
            type=FindingType.KINETIC_ANOMALY,
            severity=Severity.INFO,
            tier=Tier(tier),
            summary=f"{prefix}{metric_id} skipped on {run_label}: {reason}.",
            confidence=0.5,
            extracted_via=ExtractedVia.DETERMINISTIC,
            evidence_strength=EvidenceStrength(
                n_observations=0, n_independent_runs=0,
            ),
            evidence_observation_ids=obs_ids,
            variables_involved=[],
            run_ids=[run_id] if run_id is not None else [],
            statistics={
                "pattern_kind": pattern_kind,
                "metric_id": metric_id,
                "tier": tier,
                "reason": reason,
            },
        )


def _infer_variables(
    entry: CatalogEntry, bundle: _BundleView, run_id: str | None
) -> list[str]:
    """Best-effort: which variables did the toolkit actually consume?

    Adapters that know the truth pass `_variables_used` in stats; this
    fallback walks `entry.required_inputs` and reports which were
    present in the bundle for that run.
    """
    if run_id is None:
        return [spec.variable for spec in entry.required_inputs]
    available = bundle.variables_for(run_id)
    out: list[str] = []
    for spec in entry.required_inputs:
        if spec.variable in available:
            out.append(spec.variable)
            continue
        for proxy in spec.accepted_proxies:
            if proxy in available:
                out.append(proxy)
                break
    return out


def _collect_observation_ids(
    bundle: _BundleView, run_ids: list[str], variables: list[str]
) -> list[str]:
    out: list[str] = []
    for run_id in run_ids:
        for var in variables:
            t = bundle.trajectory(run_id, var)
            if t is not None:
                out.extend(t.source_observation_ids)
    return out


def _severity_for(metric_id: str, stats: dict) -> Severity:
    """Map raw computed values to a severity label. The catalog itself
    doesn't carry severity (a number isn't 'bad' on its own); we infer
    from named flags in the stats output.

    Adapters can override by setting `_severity_hint` in stats.
    """
    hint = stats.get("_severity_hint")
    if isinstance(hint, str):
        try:
            return Severity(hint)
        except ValueError:
            pass
    # Default: minor (computed metrics are observations, not anomalies).
    return Severity.MINOR


def _summary_for(metric_id: str, run_id: str | None, stats: dict) -> str:
    """Adapter-driven if `_summary` is set in stats; otherwise generic."""
    explicit = stats.get("_summary")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    run_label = run_id if run_id is not None else "all runs"
    return f"{metric_id} computed on {run_label}."
