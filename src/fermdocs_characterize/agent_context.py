"""AgentContext: the cacheable prompt prefix every downstream agent reads.

Goal: produce a small, structured object that captures (a) what the
experiment is, (b) what we know about its data posture, and (c) which
findings deserve attention -- without re-deriving any of it from raw
artifacts at agent-call time.

Design rules:
  - On-demand. Build per agent call. Never persisted.
  - Holds raw inputs only; rollups (n_findings_by_severity, top_findings)
    are computed at serialize time.
  - Token-budgeted. serialize_for_agent() asserts a max-tokens cap on
    fixtures and gracefully truncates oversized inputs at runtime.
  - Pure projection. AgentContext is built from (dossier, output) and
    nothing else. No DB hits, no LLM calls.

Layered identity surfaces directly:
  - process.observed.organism  -> always carried when known
  - process.registered.process_id  -> only on registry hit

This is what gives downstream agents a stable prompt prefix: the same
input produces byte-identical AgentContext, so prompt caching works.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from fermdocs_characterize.flags import ProcessFlag, compute_flags
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    Finding,
    Severity,
)
from fermdocs_characterize.specs import DictSpecsProvider, SpecsProvider
from fermdocs_characterize.views.summary import Summary, build_summary
from fermdocs_characterize.views.trajectories import build_trajectories

_log = logging.getLogger(__name__)

# Approximate token-to-character ratio. Real tokenizers vary; we err on the
# safe side so the agent prompt never silently exceeds budget.
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 1500
MAX_TOP_FINDINGS = 10
MAX_EVIDENCE_SPAN_CHARS = 200


class AgentContext(BaseModel):
    """Compact prompt prefix carrying everything an agent needs about a run.

    Intentionally hand-curated and small: ~1500 tokens after serialization.
    Held as raw fields; rollups happen in serialize_for_agent so the same
    object can serve different prompt formats without re-derivation drift.
    """

    # Identity (verbatim from dossier)
    process: dict[str, Any]

    # Provenance posture (from dossier.ingestion_summary + meta)
    schema_version: str
    extractor_version: str | None = None
    n_runs: int = 0
    n_observations: int = 0
    time_range_h: tuple[float, float] | None = None

    # Coverage (from summary)
    variables_with_specs: list[str] = Field(default_factory=list)
    variables_without_specs: list[str] = Field(default_factory=list)

    # Flags (from flags.compute_flags)
    flags: list[ProcessFlag] = Field(default_factory=list)

    # Findings raw refs; severity rollup computed at serialize time
    finding_ids: list[str] = Field(default_factory=list)


def build_agent_context(
    dossier: dict[str, Any],
    output: CharacterizationOutput,
    *,
    specs_provider: SpecsProvider | None = None,
) -> AgentContext:
    """Project (dossier, output) into an AgentContext.

    Pure function. specs_provider defaults match the CharacterizationPipeline's
    resolution: schema-with-overrides when the schema is loadable, falling
    back to dossier-only specs for offline tests / fixtures.
    """
    if specs_provider is not None:
        specs = specs_provider
    else:
        try:
            from fermdocs.domain.golden_schema import load_schema

            schema = load_schema()
            specs = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
        except Exception:
            specs = DictSpecsProvider.from_dossier(dossier)
    summary = build_summary(dossier, specs)
    trajectories = build_trajectories(summary, dossier)

    process = (dossier.get("experiment") or {}).get("process") or {}
    process = _strip_unmatched_registered_rationale(process)
    ingestion_summary = dossier.get("ingestion_summary") or {}

    schema_version = (
        output.meta.schema_version
        if output.meta and output.meta.schema_version
        else ingestion_summary.get("schema_version", "unknown")
    )

    return AgentContext(
        process=process,
        schema_version=schema_version,
        extractor_version=ingestion_summary.get("extractor_version"),
        n_runs=len(summary.run_ids),
        n_observations=len(summary.rows),
        time_range_h=_time_range(summary),
        variables_with_specs=[
            v
            for v in summary.variables
            if any(
                r.variable == v
                and r.expected is not None
                and r.expected_std_dev is not None
                for r in summary.rows
            )
        ],
        variables_without_specs=[
            v
            for v in summary.variables
            if not any(
                r.variable == v
                and r.expected is not None
                and r.expected_std_dev is not None
                for r in summary.rows
            )
        ],
        flags=compute_flags(dossier, summary, trajectories),
        finding_ids=[f.finding_id for f in output.findings],
    )


def serialize_for_agent(
    ctx: AgentContext,
    output: CharacterizationOutput,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Render AgentContext + finding rollups as JSON, bounded by max_tokens.

    Truncation policy: if the serialized blob exceeds the budget, drop the
    lowest-severity findings first until it fits. Logs a single warning per
    truncation. Never silently exceeds the budget.

    The returned string is what downstream agents receive as their primary
    context. The shape (top-level keys, ordering) is stable so prompt-cache
    prefixes hit.
    """
    findings_by_id = {f.finding_id: f for f in output.findings}
    ranked = _rank_finding_ids(ctx.finding_ids, findings_by_id)

    # Start with all top findings; truncate until we fit.
    n = min(len(ranked), MAX_TOP_FINDINGS)
    while True:
        top_ids = ranked[:n]
        blob = _build_blob(ctx, output, top_ids, findings_by_id)
        rendered = json.dumps(blob, ensure_ascii=False, sort_keys=False)
        approx_tokens = len(rendered) // CHARS_PER_TOKEN
        if approx_tokens <= max_tokens or n == 0:
            break
        n -= 1

    if n < min(len(ranked), MAX_TOP_FINDINGS):
        _log.warning(
            "agent_context: truncated top_findings from %d to %d to fit "
            "budget of %d tokens (~%d chars)",
            min(len(ranked), MAX_TOP_FINDINGS),
            n,
            max_tokens,
            max_tokens * CHARS_PER_TOKEN,
        )

    return rendered


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _strip_unmatched_registered_rationale(process: dict[str, Any]) -> dict[str, Any]:
    """Drop the LLM-written rationale when no registered process matched.

    When the identity extractor finds no registry entry that matches the
    observed organism, it still writes a rationale string into
    `process.registered.rationale` explaining what it compared against
    (e.g. "S. cerevisiae does not match the registry entry for
    Penicillium chrysogenum"). That string then travels into every
    downstream agent's prompt prefix as part of AgentContext.process,
    where it hijacks salience — agents read it as "the reference frame
    for this experiment is Penicillium" rather than "no useful reference
    in registry."

    The UNKNOWN_PROCESS flag is the routing signal; the rationale text
    is what biases the agent's framing. We strip the rationale (and the
    null process_id stays — downstream code that reads .get("process_id")
    keeps working) so the agent gets the routing signal without the
    misleading comparison string.

    No-op when registered.process_id is non-null (a real match exists,
    rationale is informative).
    """
    if not isinstance(process, dict):
        return process
    registered = process.get("registered")
    if not isinstance(registered, dict):
        return process
    if registered.get("process_id"):
        return process
    # Unmatched. Strip the rationale, keep everything else.
    cleaned_registered = {k: v for k, v in registered.items() if k != "rationale"}
    return {**process, "registered": cleaned_registered}


def _time_range(summary: Summary) -> tuple[float, float] | None:
    if not summary.rows:
        return None
    times = [r.time for r in summary.rows if r.time is not None]
    if not times:
        return None
    return (min(times), max(times))


def _severity_rank(s: Severity) -> int:
    return {
        Severity.CRITICAL: 4,
        Severity.MAJOR: 3,
        Severity.MINOR: 2,
        Severity.INFO: 1,
    }[s]


def _rank_finding_ids(
    finding_ids: list[str], by_id: dict[str, Finding]
) -> list[str]:
    """Sort findings for the agent prefix.

    Order:
      1. Severity desc (critical > major > minor > info).
      2. Within severity: aggregated rollups before per-row findings.
         A rollup carrying N=2242 violations covers more variables and
         carries strictly more information density than a per-row finding
         that flags one timestep. Without this nudge, a single high-sigma
         per-row finding crowds out N rollups for other variables.
      3. Tiebreaker: natural id order (the pipeline already pre-sorted by
         sigma desc / severity, so this is stable).
    """
    def _key(fid: str) -> tuple:
        if fid not in by_id:
            return (0, 1, fid)  # missing → lowest priority
        f = by_id[fid]
        is_aggregated = bool(f.statistics.get("aggregated"))
        return (
            -_severity_rank(f.severity),
            0 if is_aggregated else 1,  # aggregated wins within severity
            fid,
        )

    return sorted(finding_ids, key=_key)


def _severity_rollup(
    finding_ids: list[str], by_id: dict[str, Finding]
) -> dict[str, int]:
    counts: dict[str, int] = {s.value: 0 for s in Severity}
    for fid in finding_ids:
        f = by_id.get(fid)
        if f is None:
            continue
        counts[f.severity.value] += 1
    return counts


def _finding_summary(finding: Finding) -> dict[str, Any]:
    """Compact rendering of a finding for the prompt blob.

    Includes enough numeric context that the diagnosis agent can triage by
    magnitude without burning a tool call on every finding (~50 extra tokens
    per finding, total stays under 1500-token budget on existing fixtures).

    Kept compact: full evidence_observation_ids list collapsed to first id;
    statistics dict carried as-is (caller controls cardinality).
    """
    first_evidence = (
        finding.evidence_observation_ids[0]
        if finding.evidence_observation_ids
        else None
    )
    return {
        "id": finding.finding_id,
        "type": finding.type.value,
        "severity": finding.severity.value,
        "summary": finding.summary[:MAX_EVIDENCE_SPAN_CHARS],
        "confidence": round(finding.confidence, 3),
        "variables": finding.variables_involved,
        "caveats": finding.caveats[:3],
        "statistics": finding.statistics,
        "evidence_observation_id": first_evidence,
    }


def _build_blob(
    ctx: AgentContext,
    output: CharacterizationOutput,
    top_finding_ids: list[str],
    by_id: dict[str, Finding],
) -> dict[str, Any]:
    """Assemble the final dict that gets serialized.

    Stable key order so prompt-cache prefixes are byte-stable for matching
    inputs.
    """
    return {
        "process": ctx.process,
        "posture": {
            "schema_version": ctx.schema_version,
            "extractor_version": ctx.extractor_version,
            "n_runs": ctx.n_runs,
            "n_observations": ctx.n_observations,
            "time_range_h": list(ctx.time_range_h) if ctx.time_range_h else None,
        },
        "coverage": {
            "variables_with_specs": ctx.variables_with_specs,
            "variables_without_specs": ctx.variables_without_specs,
        },
        "flags": [f.value for f in ctx.flags],
        "findings": {
            "n_total": len(ctx.finding_ids),
            "by_severity": _severity_rollup(ctx.finding_ids, by_id),
            "top": [_finding_summary(by_id[fid]) for fid in top_finding_ids if fid in by_id],
        },
    }
