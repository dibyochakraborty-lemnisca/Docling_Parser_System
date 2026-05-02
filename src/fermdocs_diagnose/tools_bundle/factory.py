"""Tool registry for the diagnosis ReAct loop, curried over a bundle.

Tools are exposed via `DiagnosisToolBundle`. Every method:
  - Takes simple JSON-able args (str / list[str] / int / dict)
  - Returns a JSON-able dict (never raises across the agent boundary)
  - Self-documents via `description` for the system prompt catalog

State machine (plan §3.1):

    RUNNING ──submit_diagnosis()──▶ SUBMITTED ──finalize()──▶ DONE
       │                                  │
       └─ budget exhausted ──────────────▶┘
       any tool call after SUBMITTED ▶ {"error": "already_finalized"}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from fermdocs.bundle import BundleReader
from fermdocs.domain.process_priors import (
    ProcessPriors,
    cached_priors,
    resolve_priors,
)
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.specs import DictSpecsProvider, SpecsProvider
from fermdocs_diagnose.audit.trace_writer import TraceWriter
from fermdocs_diagnose.tools_bundle.execute_python import (
    PYTHON_DEFAULT_TIMEOUT,
    ExecutePythonResult,
    execute_python,
)


class ToolError(Exception):
    """Raised internally when a tool can't fulfill its contract.

    The dispatcher converts these into `{"error": ...}` payloads so the agent
    sees a clean tool result.
    """


def _sanitize_json(obj: Any) -> Any:
    """Replace NaN/Inf with None; pandas frames frequently dump NaN that
    Pydantic or downstream JSON consumers reject."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


@dataclass
class _AgentState:
    """Single-call state shared across tool invocations."""

    diagnosis_payload: dict | None = None  # what submit_diagnosis received
    flags: dict[str, bool] = field(default_factory=dict)
    submitted: bool = False
    tool_calls: int = 0


@dataclass
class DiagnosisToolBundle:
    """The set of tools the diagnosis agent can call.

    Construct via `make_diagnosis_tools(reader, ...)`. Each method's docstring
    is the agent-facing description used to build the system prompt catalog.
    """

    reader: BundleReader
    specs: SpecsProvider
    upstream: CharacterizationOutput
    trace_writer: TraceWriter | None = None
    priors: ProcessPriors | None = None
    state: _AgentState = field(default_factory=_AgentState)

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _gate(self, tool_name: str) -> dict | None:
        """Return an error payload if the agent has already submitted."""
        self.state.tool_calls += 1
        if self.state.submitted and tool_name != "submit_diagnosis":
            return {"error": "already_finalized", "tool": tool_name}
        return None

    # ------------------------------------------------------------------
    # Read-side tools
    # ------------------------------------------------------------------

    def list_runs(self) -> dict:
        """List all run_ids in this bundle.

        Cost: Low. Returns {"run_ids": [...]}. Always succeeds.
        """
        gated = self._gate("list_runs")
        if gated:
            return gated
        return {"run_ids": list(self.reader.meta.run_ids)}

    def get_meta(self) -> dict:
        """Return bundle metadata: schema versions, run_ids, model_labels, flags.

        Cost: Low. Use this once at the start of analysis to know what you're
        looking at (organism, schema version, model used for characterization).
        """
        gated = self._gate("get_meta")
        if gated:
            return gated
        bundle_dir = self.reader.dir
        obs_path = bundle_dir / "characterization" / "observations.csv"
        return {
            "bundle_id": self.reader.meta.bundle_id,
            "bundle_schema_version": self.reader.meta.bundle_schema_version,
            "golden_schema_version": self.reader.meta.golden_schema_version,
            "pipeline_version": self.reader.meta.pipeline_version,
            "run_ids": list(self.reader.meta.run_ids),
            "model_labels": dict(self.reader.meta.model_labels),
            "flags": dict(self.reader.meta.flags),
            "bundle_dir": str(bundle_dir),
            "observations_csv_path": str(obs_path) if obs_path.exists() else None,
            "observations_csv_columns": [
                "run_id", "variable", "time_h", "value", "imputed", "unit",
            ],
            "process_priors_version": self.priors.version if self.priors is not None else None,
            "process_priors_organisms": (
                [o.name for o in self.priors.organisms] if self.priors is not None else []
            ),
            # Plan B Stage 3: surface narrative count + tags so the agent
            # knows whether to call get_narrative_observations.
            "narrative_observations_count": len(self.upstream.narrative_observations),
            "narrative_observation_tags": sorted(
                {getattr(n.tag, "value", str(n.tag)) for n in self.upstream.narrative_observations}
            ),
        }

    def get_findings(
        self,
        *,
        finding_id: str | None = None,
        run_id: str | None = None,
        variable: str | None = None,
        severity: str | None = None,
        tier: str | None = None,
        limit: int = 100,
    ) -> dict:
        """Filtered access to deterministic findings from characterization.

        Findings are RANKED ranges/aggregates produced upstream — they may
        misframe the problem (a clean run can produce thousands of "violations"
        if specs are setpoints not bounds). Use them as evidence, not as the
        spine of your analysis.

        Cost: Low. Returns {"findings": [...], "total": N, "truncated": bool}.
        """
        gated = self._gate("get_findings")
        if gated:
            return gated
        all_findings = list(self.upstream.findings)

        def _match(f: Any) -> bool:
            if finding_id and f.finding_id != finding_id:
                return False
            if run_id and run_id not in f.run_ids:
                return False
            if variable and variable not in f.variables_involved:
                return False
            if severity and f.severity.value != severity:
                return False
            if tier and getattr(f.tier, "value", str(f.tier)) != tier:
                return False
            return True

        matched = [f for f in all_findings if _match(f)]
        truncated = len(matched) > limit
        sliced = matched[:limit]
        return {
            "findings": [_sanitize_json(f.model_dump(mode="json")) for f in sliced],
            "total": len(matched),
            "truncated": truncated,
        }

    def get_narrative_observations(
        self,
        *,
        run_id: str | None = None,
        tag: str | None = None,
        variable: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Return prose insights extracted from the source document.

        Cost: Low. Returns {"observations": [...], "total": N, "truncated": bool,
        "tags_present": [...]}.

        These are direct statements from the report author / operator —
        closure events ("white cells observed"), interventions ("IPM added"),
        deviations, conclusions. Treat them as primary evidence: if the report
        says cells died, they died, regardless of what biomass numbers show.

        Filter args:
            run_id: only observations attributed to that run (e.g. "BATCH-01")
            tag: closure_event | deviation | intervention | observation |
                 conclusion | protocol_note
            variable: only observations whose affected_variables include this name
            limit: hard cap on returned items (default 50)

        You SHOULD call this once at the start of analysis on any bundle that
        has narrative observations. Cite narrative_ids in cited_narrative_ids
        on any claim grounded in prose.
        """
        gated = self._gate("get_narrative_observations")
        if gated:
            return gated
        all_obs = list(self.upstream.narrative_observations)

        def _match(n: Any) -> bool:
            if run_id and n.run_id != run_id:
                return False
            if tag and getattr(n.tag, "value", str(n.tag)) != tag:
                return False
            if variable and variable not in n.affected_variables:
                return False
            return True

        matched = [n for n in all_obs if _match(n)]
        truncated = len(matched) > limit
        sliced = matched[:limit]
        tags_present = sorted({getattr(n.tag, "value", str(n.tag)) for n in all_obs})
        return {
            "observations": [_sanitize_json(n.model_dump(mode="json")) for n in sliced],
            "total": len(matched),
            "truncated": truncated,
            "tags_present": tags_present,
        }

    def get_specs(self, variable: str) -> dict:
        """Return the schema spec (nominal, std_dev, unit, source) for a variable.

        Cost: Low. Returns {"variable": str, "spec": {...}|null}.
        Specs are SETPOINTS, not trajectory bounds — useful for unit context,
        not as anomaly thresholds for fed-batch processes.
        """
        gated = self._gate("get_specs")
        if gated:
            return gated
        spec = self.specs.get(variable)
        if spec is None:
            return {"variable": variable, "spec": None}
        from dataclasses import asdict, is_dataclass

        if is_dataclass(spec):
            return {"variable": variable, "spec": asdict(spec)}
        if hasattr(spec, "model_dump"):
            return {"variable": variable, "spec": spec.model_dump(mode="json")}
        return {"variable": variable, "spec": dict(spec) if isinstance(spec, dict) else str(spec)}

    def get_priors(
        self,
        organism: str | None = None,
        process_family: str | None = None,
        variable: str | None = None,
    ) -> dict:
        """Return organism × process-family expected ranges (literature-sourced).

        Cost: Low. Returns {"priors": [...], "n": int, "matched_organism": str|None,
        "matched_process_family": str|None, "available_organisms": [...]}.

        Use this BEFORE claiming a value is anomalous on a single-run dossier.
        Without priors, single-run reasoning falls back to schema_only and
        confidence is capped lower. The bundle's organism comes from get_meta()
        and the dossier — pass it through to get organism-specific ranges.

        Each prior carries a `source` citation (e.g. "Verduyn 1991"). Cite the
        source in your claim's summary when grounding a finding on a prior.

        When no priors match (uncovered organism), returns an empty list and
        names the available organisms so you can see what's loaded.
        """
        gated = self._gate("get_priors")
        if gated:
            return gated
        if self.priors is None:
            return {
                "priors": [],
                "n": 0,
                "matched_organism": None,
                "matched_process_family": None,
                "available_organisms": [],
                "note": "No priors loaded for this bundle.",
            }
        rows = resolve_priors(
            self.priors,
            organism=organism,
            process_family=process_family,
            variable=variable,
        )
        # Identify which organism actually matched (substring resolution
        # may pick the canonical name even when the query was an alias).
        matched_org = rows[0].organism if rows else None
        matched_fam = rows[0].process_family if rows else None
        return {
            "priors": [r.to_dict() for r in rows],
            "n": len(rows),
            "matched_organism": matched_org,
            "matched_process_family": matched_fam,
            "available_organisms": [o.name for o in self.priors.organisms],
        }

    def get_timecourse(
        self,
        run_id: str,
        variable: str,
        *,
        time_range_h: list[float] | None = None,
        max_points: int = 100,
    ) -> dict:
        """Return a (run_id, variable) trajectory slice.

        Cost: Low. Returns {"run_id", "variable", "unit", "time_grid",
        "values", "imputation_flags", "n_points", "truncated"}.

        Missing values come back as null (JSON-safe). For full numerical work,
        prefer `execute_python` and load the bundle's parquet directly.
        """
        gated = self._gate("get_timecourse")
        if gated:
            return gated
        for t in self.upstream.trajectories:
            if t.run_id == run_id and t.variable == variable:
                grid = t.time_grid
                vals = t.values
                flags = t.imputation_flags
                if time_range_h and len(time_range_h) == 2:
                    lo, hi = float(time_range_h[0]), float(time_range_h[1])
                    keep = [i for i, x in enumerate(grid) if lo <= x <= hi]
                    grid = [grid[i] for i in keep]
                    vals = [vals[i] for i in keep]
                    flags = [flags[i] for i in keep]
                truncated = len(grid) > max_points
                if truncated:
                    grid = grid[:max_points]
                    vals = vals[:max_points]
                    flags = flags[:max_points]
                return _sanitize_json(
                    {
                        "run_id": run_id,
                        "variable": variable,
                        "unit": t.unit,
                        "time_grid": grid,
                        "values": vals,
                        "imputation_flags": flags,
                        "n_points": len(grid),
                        "truncated": truncated,
                    }
                )
        return {
            "error": "trajectory_not_found",
            "run_id": run_id,
            "variable": variable,
            "available": [
                {"run_id": tt.run_id, "variable": tt.variable}
                for tt in self.upstream.trajectories
            ],
        }

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def execute_python(self, code: str, timeout: int = PYTHON_DEFAULT_TIMEOUT) -> dict:
        """Run Python in a sandboxed subprocess (pandas, numpy, scipy, sklearn,
        plotly available; cwd is project root so `from fermdocs...` imports work).

        Cost: HIGH. Use when fetch tools can't answer your question — derived
        rates, cross-batch comparison, anomaly detection on shapes, etc.
        Use print() for output; only stdout is returned.

        Returns {"stdout", "stderr", "returncode", "timed_out", "duration_ms"}.
        Stdout/stderr capped at 50KB; full text persisted to audit/.
        """
        gated = self._gate("execute_python")
        if gated:
            return gated
        result: ExecutePythonResult = execute_python(
            code, timeout=timeout, trace_writer=self.trace_writer
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
        }

    # ------------------------------------------------------------------
    # Terminator
    # ------------------------------------------------------------------

    def submit_diagnosis(self, payload: dict) -> dict:
        """Finalize the diagnosis. Idempotent on identical payload; second
        call with a different payload returns {"error": "diagnosis_already_submitted"}.

        Cost: n/a (terminator). The runtime validates the payload and writes
        diagnosis/diagnosis.json. After this call, no other tool will execute.
        """
        self.state.tool_calls += 1
        if self.state.submitted:
            if self.state.diagnosis_payload == payload:
                return {"ok": True, "idempotent": True}
            return {"error": "diagnosis_already_submitted"}
        self.state.diagnosis_payload = payload
        self.state.submitted = True
        return {"ok": True}


def make_diagnosis_tools(
    reader: BundleReader,
    upstream: CharacterizationOutput,
    *,
    specs: SpecsProvider | None = None,
    dossier: dict | None = None,
    trace_writer: TraceWriter | None = None,
    priors: ProcessPriors | None = None,
    load_default_priors: bool = True,
) -> DiagnosisToolBundle:
    """Build a tool bundle curried over `reader`.

    Args:
      specs: defaults to DictSpecsProvider.from_dossier(dossier) when dossier
        is provided, else an empty provider.
      priors: explicit ProcessPriors instance. When omitted and
        load_default_priors=True (the default), loads the shipped
        process_priors.yaml. Set load_default_priors=False to run priors-less
        (used by tests that need to exercise the no-priors path).
    """
    if specs is None:
        if dossier is None:
            try:
                dossier = reader.get_dossier()
            except FileNotFoundError:
                dossier = {}
        specs = DictSpecsProvider.from_dossier(dossier)
    if priors is None and load_default_priors:
        try:
            priors = cached_priors()
        except (OSError, ValueError):
            priors = None
    return DiagnosisToolBundle(
        reader=reader,
        upstream=upstream,
        specs=specs,
        trace_writer=trace_writer,
        priors=priors,
    )
