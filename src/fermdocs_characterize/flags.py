"""ProcessFlag: closed-vocabulary signals about a run's data posture.

Flags are deterministic summaries downstream agents read alongside the
findings. Each flag's emission rule, threshold, and rationale lives in its
docstring on the enum so the rule and the value ship together.

The flag set is intentionally short and high-signal. Agents prompt-tune
behavior on the flag list without re-reading the whole dossier.

Adding a flag:
  1. New enum value with full docstring (rule + threshold + rationale)
  2. New branch in compute_flags()
  3. One test per emission boundary in tests/unit/test_flags.py

Reading a flag:
  - Treat as a routing signal, not a finding. UNKNOWN_PROCESS does not mean
    the run is bad; it means the agent should back off process-specific
    priors. SPARSE_DATA does not mean the data is wrong; it means
    statistical claims need wider error bars.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fermdocs_characterize.schema import Trajectory
from fermdocs_characterize.views.summary import Summary


class ProcessFlag(str, Enum):
    """Closed vocabulary of data-posture flags.

    Each value's docstring documents the exact firing rule. Validators
    reject anything not in this list.
    """

    STALE_SCHEMA_OBSERVATIONS = "stale_schema_observations"
    """Fires when dossier.ingestion_summary.stale_schema_versions is non-empty.

    Rule: any observation in the experiment was extracted under a schema
    version different from the currently-loaded schema.

    Rationale: column names or unit semantics may have shifted. Findings
    derived from old observations are suspect; agents should flag rather
    than reason aggressively.
    """

    LOW_QUALITY_TRAJECTORY = "low_quality_trajectory"
    """Fires when any Trajectory has quality < 0.8.

    Rule: at least one (run_id, variable) trajectory has more than 20% of
    its grid points imputed or missing.

    Rationale: derivative-based reasoning (rates, slopes, plateau breaks)
    is unreliable when most of the curve is interpolated. Range checks
    against single observations remain valid.
    """

    SPARSE_DATA = "sparse_data"
    """Fires when summary.rows < 20 OR golden_coverage_percent < 50.

    Rule: fewer than 20 numeric observations across the whole experiment,
    or fewer than half of the golden columns have any observation at all.

    Rationale: pattern detection has a noise floor; a handful of
    measurements is enough for sanity checks but not for cohort-level
    claims.
    """

    MIXED_RUNS = "mixed_runs"
    """Fires when len(summary.run_ids) > 1.

    Rule: the dossier carries observations from more than one run_id.

    Rationale: cohort comparisons require the agent to know it's looking
    at multiple runs. Same-experiment-different-run patterns get reasoned
    about differently than within-run patterns.
    """

    UNKNOWN_PROCESS = "unknown_process"
    """Fires when dossier.experiment.process.registered.provenance == 'unknown'.

    Rule: the registered-process layer is UNKNOWN -- the experiment did
    not match a registry entry. The observed layer (organism, product,
    scale) may still be populated; that's checked separately.

    Rationale: identity is the strongest prior on what's normal. Without
    a registered recipe, agents should not invoke process-specific
    expectations (e.g. expected DO crash patterns). They can still reason
    from observed surface facts and from the schema's nominal/std_dev.
    """

    UNKNOWN_ORGANISM = "unknown_organism"
    """Fires when dossier.experiment.process.observed.organism is null/empty.

    Rule: even the surface-fact layer has no organism.

    Rationale: this is the harshest agent-context flag. With no organism,
    agents have neither a recipe nor biology to reason from. Findings
    should rely entirely on schema specs and explicit values.
    """

    SPECS_MOSTLY_MISSING = "specs_mostly_missing"
    """Fires when more than 50% of summary.variables have no spec.

    Rule: of the variables that produced at least one observation, more
    than half have no nominal+std_dev pair in the loaded golden schema.

    Rationale: range-violation reasoning needs specs. When most are
    missing, the output is heavy on open_questions and light on findings;
    agents should skew toward asking, not telling.
    """


# Thresholds are module-level so tests can reference them directly.
LOW_QUALITY_THRESHOLD = 0.8
SPARSE_ROWS_THRESHOLD = 20
SPARSE_COVERAGE_PCT_THRESHOLD = 50
SPECS_MISSING_PCT_THRESHOLD = 0.5


def compute_flags(
    dossier: dict[str, Any],
    summary: Summary,
    trajectories: list[Trajectory],
) -> list[ProcessFlag]:
    """Apply every emission rule and return a sorted, deduplicated list.

    Pure function: same inputs -> same output. Order is deterministic
    (alphabetical on the string value) so the agent context blob is
    cache-friendly.
    """
    flags: set[ProcessFlag] = set()

    # STALE_SCHEMA_OBSERVATIONS
    ingestion_summary = dossier.get("ingestion_summary") or {}
    if ingestion_summary.get("stale_schema_versions"):
        flags.add(ProcessFlag.STALE_SCHEMA_OBSERVATIONS)

    # LOW_QUALITY_TRAJECTORY
    for traj in trajectories:
        if traj.quality < LOW_QUALITY_THRESHOLD:
            flags.add(ProcessFlag.LOW_QUALITY_TRAJECTORY)
            break

    # SPARSE_DATA
    coverage_pct = ingestion_summary.get("golden_coverage_percent", 0)
    if (
        len(summary.rows) < SPARSE_ROWS_THRESHOLD
        or coverage_pct < SPARSE_COVERAGE_PCT_THRESHOLD
    ):
        flags.add(ProcessFlag.SPARSE_DATA)

    # MIXED_RUNS
    if len(summary.run_ids) > 1:
        flags.add(ProcessFlag.MIXED_RUNS)

    # UNKNOWN_PROCESS / UNKNOWN_ORGANISM
    process = (dossier.get("experiment") or {}).get("process") or {}
    registered = process.get("registered") or {}
    observed = process.get("observed") or {}
    if registered.get("provenance") == "unknown":
        flags.add(ProcessFlag.UNKNOWN_PROCESS)
    if not (observed.get("organism") or "").strip():
        flags.add(ProcessFlag.UNKNOWN_ORGANISM)

    # SPECS_MOSTLY_MISSING
    if summary.variables:
        without_specs = [
            v
            for v in summary.variables
            if not _variable_has_spec(v, summary)
        ]
        ratio = len(without_specs) / len(summary.variables)
        if ratio > SPECS_MISSING_PCT_THRESHOLD:
            flags.add(ProcessFlag.SPECS_MOSTLY_MISSING)

    return sorted(flags, key=lambda f: f.value)


def _variable_has_spec(variable: str, summary: Summary) -> bool:
    """A variable counts as having a spec when at least one of its rows
    carries non-None expected and expected_std_dev.
    """
    for row in summary.rows:
        if (
            row.variable == variable
            and row.expected is not None
            and row.expected_std_dev is not None
        ):
            return True
    return False
