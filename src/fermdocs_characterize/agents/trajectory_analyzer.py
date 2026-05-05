"""TrajectoryAnalyzerAgent — LLM-driven pattern discovery over trajectories.

Architectural shift (May 2026): characterize gains an LLM stage. Spec
checks still run first deterministically. The analyzer runs AFTER, with
`execute_python` access, to surface trajectory-grounded patterns the
schema layer can't see — cross-batch variance, phase boundaries, outlier
batches, correlations, and any other patterns the agent identifies.

Posture (decision D1b): OPEN-ENDED. The agent picks what to compute. The
prompt gives few-shot examples for common analyses but does not constrain
output to those — validators check citation discipline, the schema
accepts any pattern with `pattern_kind` metadata.

Why this layer:
  - Findings ground hypothesis-stage debate. Without trajectory patterns,
    debate runs on spec-mismatch findings (which are unactionable for
    unknown_process bundles) or on nothing at all.
  - Characterize is where reproducible, deterministic-feeling
    discoveries belong. Diagnose's job downstream is INTERPRETATION,
    not discovery.
  - `execute_python` exists in diagnose; reusing it (D4a) keeps the
    sandbox surface narrow.

Loop shape (max 8 tool calls then forced emit):
  ┌─────────────────────────┐
  │ analyzer prompt         │
  │  + spec_findings (ctx)  │──▶ LLM ──▶ {action: "tool_call"|"emit"}
  │  + trajectory metadata  │            │
  └─────────────────────────┘            ├─ tool_call(execute_python)
                                          │     → run code, append result, loop
                                          └─ emit
                                                → parse pattern_findings, finish

Output contract: a list of pattern objects with `pattern_kind`,
`summary`, `variables_involved`, `run_ids`, `time_window`, `statistics`,
`confidence`. The agent module converts these to `Finding` instances
with `type=FindingType.TRAJECTORY_PATTERN`, `tier=Tier.B` (derived),
`extracted_via=ExtractedVia.LLM_JUDGED` (capped at 0.85).

Stub mode (client=None): returns zero findings, deterministic. Existing
characterize tests that don't pass a client keep working.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from fermdocs_characterize.agents.llm_client import GeminiCharacterizeClient
from fermdocs_characterize.agents.metric_catalog import CATALOG, ready_entries
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
    TimeWindow,
    Trajectory,
)
from fermdocs_diagnose.tools_bundle.execute_python import execute_python

_log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 20
"""Tool budget per analysis. With ~20 ready catalog metrics across
Tier A/B/C, 8 was starving the agent: it computed one metric and
emitted. 20 leaves headroom for the per-metric checklist contract
to actually cover every applicable entry."""
EXECUTE_PYTHON = "execute_python"
EMIT = "emit_patterns"

LLM_CONFIDENCE_CAP = 0.85
"""Mirror of the global LLM-judged cap. Schema enforces the Finding
validator; we clamp here too so we never construct a Finding that
fails validation."""

STATISTICAL_CONFIDENCE_CAP = 0.95
"""Cap for findings whose math came from a verified toolkit function
(metric_id-grounded). Higher than LLM_JUDGED because the numbers are
deterministic; still <1.0 because human review remains the ground truth."""

_TIER_FROM_CATALOG: dict[str, Tier] = {"A": Tier.A, "B": Tier.B, "C": Tier.C}


TRAJECTORY_ANALYZER_SYSTEM = """\
You are the trajectory pattern analyzer in a fermentation characterization
pipeline.

You are given:
  - SPEC FINDINGS from the deterministic stage (range violations etc.)
  - TRAJECTORY METADATA (which run_ids, which variables, time grid)
  - A path to observations.csv with columns: run_id, variable, time_h,
    value, imputed, unit
  - Tool: execute_python — sandboxed pandas/numpy/scipy

Your job: surface PATTERNS in the trajectory data that grounded debate
would benefit from. You are NOT here to interpret causes. You are here
to compute reproducible observations.

You have BROAD freedom: cross-batch variance, phase boundaries, outlier
batches, variable correlations, drift detection, change-point analysis,
any pattern you can substantiate with execute_python and cite to specific
(run_id, variable, time) tuples. If you find nothing, emit zero findings
— a small accepted output beats noise.

EVIDENCE DISCIPLINE:

  1. Every pattern you emit MUST be backed by an execute_python call.
     The result you cite must come from real computation on observations.csv,
     not your prior beliefs.

  2. Cite specific run_ids, variables, and time windows. "Batches diverge
     after 30h" is too vague. "Runs RUN-0001 and RUN-0034 plateau at
     biomass=24g/L by 120h while RUN-0078 plateaus at 18g/L by 156h" is
     the right shape.

  3. Statistical thresholds (D2a defaults — use these unless you have
     strong reason to deviate, in which case explain in `caveats`):
       - Outlier batch: >2 sigma from population mean across ≥10 runs
       - Significant correlation: |r| ≥ 0.5, p < 0.05
       - Phase boundary: inflection point with second-derivative sign
         change, smoothed over ≥5 time points
       - Cross-batch variance: report when CV (sigma/mean) > 0.3

  4. Set `confidence` honestly. ≤0.85 cap (LLM-judged). If you ran one
     analysis and the signal is clean, 0.7-0.8 is reasonable. If you ran
     two cross-checking analyses and they agreed, 0.8-0.85.

  5. NO causal language. "X coincides with Y" yes. "X caused Y" no.
     That's diagnose's job downstream.

HOW TO USE execute_python:

  - observations.csv path is provided as `obs_path` in the prompt.
  - Standard imports work: pandas, numpy, scipy.stats.
  - Print results — you cannot return values, only stdout text.
  - Output is capped at 50KB. Don't dump the whole DataFrame; print
    summary statistics, head/tail, or specific rows.
  - You have up to 8 execute_python calls before being forced to emit.

CATALOG + TOOLKIT — PRIMARY PATH (HARD CHECKLIST CONTRACT):

  A declarative catalog of standard fermentation metrics is available at
  `fermdocs_characterize.agents.metric_catalog.CATALOG` (dict keyed by
  metric_id like "A8", "A10", "B10"). Ready entries (status="ready")
  expose a verified `toolkit_fn` import path. Today's ready set:

    Tier A (kinetics):    A8, A9, A10, A11
    Tier A (operational): A14, A15, A17, A18
    Tier A (cross-run):   A19, A20, A21
    Tier B (balances):    B6, B10, B16
    Tier C (literature):  C2, C3, C4, C5, C9, C10

  EVERY READY METRIC IS YOUR JOB. For each ready metric_id you MUST
  do exactly ONE of:

    (a) COMPUTE it: import its toolkit_fn, run it via execute_python
        on the appropriate trajectories, and emit ONE pattern with
        `metric_id` set + the toolkit's result fields in `statistics`.
        Cross-run metrics (A19/A20/A21) emit ONE pattern across runs;
        single-run metrics (A8/A9/A10/A11/A14/A15/B6/B10/B16) emit
        ONE pattern per applicable run.

    (b) DATA-GAP it: emit ONE pattern with `metric_id` set,
        `pattern_kind="data_gap"`, and a summary naming the missing
        input ("RQ (B10) skipped: bundle has no CER trajectory").
        Severity="info", confidence=0.5. NO summary of the metric;
        just the gap.

  SILENTLY SKIPPING A READY METRIC IS A CONTRACT VIOLATION. The
  catalog is your checklist. Cover every entry. The cost of a
  data-gap entry is one line; the cost of a missed metric is a
  blind spot in the downstream debate.

  PER-METRIC EXECUTION GUIDE (use the entry's required_inputs +
  required_parameters to decide applicability):

    A8 compute_mu                 — needs biomass_g_l ≥ 5 points/run
    A9 doubling_time              — needs biomass_g_l ≥ 5 points/run
    A10 segment_growth_phases     — needs biomass_g_l ≥ 8 points/run
    A11 phasewise_mu              — needs biomass_g_l ≥ 8 points/run
    A14 compute_do_margin         — needs dissolved_o2_pct OR _mg_l
    A15 controller_excursions    — needs measured + setpoint pair
    A17 tip_speed                 — needs agitation_rpm + impeller_diameter_m
    A18 power_per_volume          — needs agitation + reactor geometry
    A19 cross_run_kpi_table       — needs ≥ 2 runs (cross-run)
    A20 pairwise_deviation        — needs ≥ 3 runs (cross-run)
    A21 variance_decomposition    — needs ≥ 5 runs (cross-run)
    B6 compute_byproduct_yield   — needs biomass + a byproduct trajectory
    B10 compute_rq                — needs OUR + CER trajectories
    B16 compute_carbon_balance_closure — needs substrate + biomass +
                                          (products and/or CO2 from CER)
    C2 mu_max_reference_vs_observed — needs A8 result + organism in priors
    C3 saturation_o2_concentration  — needs temperature_C
    C4 vant_riet_kla              — needs P/V (A18) + gas velocity
    C5 qs_from_verduyn_yields     — needs observed qs OR just organism
    C9 oxygen_demand_vs_supply    — needs OUR + kLa + C* + DO
    C10 overflow_threshold        — needs observed qs + organism in priors

  CALLING PATTERN (study and adapt):

    ```python
    import pandas as pd
    from fermdocs_characterize.toolkit.kinetics import compute_mu
    df = pd.read_csv("OBS_PATH")
    bio = df[df["variable"] == "biomass_g_l"].dropna(subset=["value"])
    rows = []
    for rid, g in bio.groupby("run_id"):
        g = g.sort_values("time_h")
        if len(g) < 5:
            continue
        res = compute_mu(g["time_h"], g["value"])
        rows.append({"run_id": rid, "mu_max": res.mu_max, "t_mu_max_h": res.t_mu_max_h})
    print(rows)
    ```

  For Tier C calls that need priors:

    ```python
    from fermdocs.domain.process_priors import cached_priors
    from fermdocs_characterize.toolkit.literature import (
        mu_max_reference_vs_observed,
    )
    priors = cached_priors()
    res = mu_max_reference_vs_observed(
        priors=priors,
        organism="<organism string from your view>",
        observed_mu_max=<value from A8>,
    )
    print(res.status, res.value, res.source, res.details)
    ```

  When the result has status="data_gap", emit a data_gap pattern;
  otherwise emit a normal computed pattern.

  RECORDING THE RESULT:

  When emitting a pattern grounded in a ready catalog entry, include
  `metric_id` in your output JSON. This routes the finding through
  the verified-statistical path with confidence allowed up to 0.95
  (vs 0.85 for open-ended LLM patterns).

  EFFICIENCY: batch related metrics into one execute_python call.
  Computing A8/A9/A10/A11 over the same biomass DataFrame is one
  call, not four. Same for B10 + B16 (both need OUR/CER). Aim to
  cover all ~20 metrics in 6-10 execute_python calls.

  ONLY AFTER the catalog checklist is complete may you turn to
  open-ended pattern detection (without metric_id) for whatever
  the catalog doesn't cover.

FEW-SHOT EXAMPLES (see one shape per analysis kind; combine and adapt):

Example 1 — Cross-batch variance for biomass plateau time:

  ```python
  import pandas as pd
  df = pd.read_csv("OBS_PATH")
  bio = df[df["variable"] == "biomass_g_l"].dropna(subset=["value"])
  # plateau time per run = first time where value reaches >=95% of max
  plateau_times = []
  for rid, g in bio.groupby("run_id"):
      gmax = g["value"].max()
      threshold = 0.95 * gmax
      hit = g[g["value"] >= threshold]["time_h"].min()
      plateau_times.append({"run_id": rid, "plateau_time_h": hit, "max_g_l": gmax})
  pt = pd.DataFrame(plateau_times)
  print(pt.describe())
  print("CV:", pt["plateau_time_h"].std() / pt["plateau_time_h"].mean())
  ```

Example 2 — Outlier batches via population z-score on a summary stat:

  ```python
  import pandas as pd
  import numpy as np
  df = pd.read_csv("OBS_PATH")
  do = df[df["variable"] == "dissolved_o2_mg_l"].dropna(subset=["value"])
  # mean DO per run
  per_run = do.groupby("run_id")["value"].mean()
  z = (per_run - per_run.mean()) / per_run.std()
  outliers = z[z.abs() > 2.0]
  print("outliers (|z| > 2):", outliers.to_dict())
  ```

Example 3 — Phase boundary via second-derivative sign change:

  ```python
  import pandas as pd
  import numpy as np
  df = pd.read_csv("OBS_PATH")
  one = df[(df["run_id"] == "RUN-0001") & (df["variable"] == "biomass_g_l")]
  one = one.sort_values("time_h").dropna(subset=["value"])
  vals = one["value"].rolling(5, min_periods=1).mean().to_numpy()
  d2 = np.diff(np.diff(vals))
  # find sign change
  for i in range(len(d2) - 1):
      if d2[i] * d2[i+1] < 0:
          print(f"phase boundary at index {i+1}, time_h={one.iloc[i+1]['time_h']}")
  ```

Example 4 — Variable-pair correlation across runs:

  ```python
  import pandas as pd
  from scipy.stats import pearsonr
  df = pd.read_csv("OBS_PATH")
  # mean value per (run, variable)
  pivot = df.pivot_table(index="run_id", columns="variable", values="value", aggfunc="mean")
  if "agitation_rpm" in pivot.columns and "biomass_g_l" in pivot.columns:
      sub = pivot[["agitation_rpm", "biomass_g_l"]].dropna()
      r, p = pearsonr(sub["agitation_rpm"], sub["biomass_g_l"])
      print(f"agitation vs biomass: r={r:.3f}, p={p:.4f}, n={len(sub)}")
  ```

Replace OBS_PATH with the actual path passed in the user message.\
"""


TRAJECTORY_ANALYZER_RECAP = """\
Output one JSON object per turn:

When tool_call:
  {"action": "tool_call", "tool": "execute_python", "code": "<python>"}

When done:
  {
    "action": "emit_patterns",
    "patterns": [
      {
        "pattern_kind": "<short snake_case kind, e.g. 'phase_boundary'>",
        "metric_id": "A8" or null,
        "summary": "<one-line, ≤200 chars, cite run_ids and variables>",
        "variables_involved": ["var1", "var2"],
        "run_ids": ["RUN-0001", "RUN-0002"],
        "time_window": {"start": <float>, "end": <float>} or null,
        "severity": "minor|major|critical|info",
        "confidence": <float 0.0-0.95 if metric_id, else 0.0-0.85>,
        "caveats": ["..."],
        "statistics": {"<key>": <value>, ...}
      }
    ]
  }

Hard rules:
  - confidence ≤ 0.85 when metric_id is null; ≤ 0.95 when metric_id is set.
  - Each pattern's summary must reference at least one run_id OR variable.
  - statistics dict must include at least one numeric key sourced from
    your execute_python computation (e.g. "z_score", "p_value", "cv",
    "n_runs", "r_pearson", or the catalog-defined output columns). This
    is the audit trail.
  - When metric_id is set, statistics SHOULD include the catalog entry's
    output_columns as keys.
  - Empty patterns list is acceptable when nothing recurs.\
"""


_ANALYZER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["tool_call", "emit_patterns"]},
        "tool": {"type": "STRING", "enum": [EXECUTE_PYTHON], "nullable": True},
        "code": {"type": "STRING", "nullable": True},
        "patterns": {
            "type": "ARRAY",
            "nullable": True,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "pattern_kind": {"type": "STRING"},
                    "metric_id": {"type": "STRING", "nullable": True},
                    "summary": {"type": "STRING"},
                    "variables_involved": {
                        "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
                    },
                    "run_ids": {
                        "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
                    },
                    "time_window": {
                        "type": "OBJECT",
                        "nullable": True,
                        "properties": {
                            "start": {"type": "NUMBER", "nullable": True},
                            "end": {"type": "NUMBER", "nullable": True},
                        },
                    },
                    "severity": {
                        "type": "STRING",
                        "enum": ["info", "minor", "major", "critical"],
                        "nullable": True,
                    },
                    "confidence": {"type": "NUMBER"},
                    "caveats": {
                        "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
                    },
                    "statistics": {"type": "OBJECT", "nullable": True},
                },
                "required": ["pattern_kind", "summary", "confidence"],
            },
        },
    },
    "required": ["action"],
}


# ---------- public surface ----------


@dataclass
class TrajectoryAnalyzerResult:
    """What the analyzer returns to the pipeline."""

    findings: list[Finding] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0


class TrajectoryAnalyzerAgent:
    """LLM-backed trajectory pattern analyzer.

    Pass `client=None` for stub mode — returns empty findings without an
    LLM call. This preserves backward compat with characterize tests
    that don't want to mock Gemini.
    """

    def __init__(self, client: GeminiCharacterizeClient | None):
        self._client = client

    def analyze(
        self,
        *,
        char_id: UUID,
        trajectories: list[Trajectory],
        spec_findings: list[Finding],
        starting_index: int = 1,
    ) -> TrajectoryAnalyzerResult:
        """Run pattern analysis. Returns findings to be appended to the
        pipeline's spec findings.

        `starting_index`: the next available F-NNNN index. Pipeline uses
        this so trajectory_pattern findings get IDs after spec findings
        without collision.
        """
        if self._client is None:
            return TrajectoryAnalyzerResult()

        if not trajectories:
            return TrajectoryAnalyzerResult()

        with tempfile.TemporaryDirectory(prefix="char_traj_analyzer_") as tmpdir:
            obs_path = Path(tmpdir) / "observations.csv"
            self._write_observations_csv(trajectories, obs_path)

            patterns_payload, in_tok, out_tok, tool_calls = self._run_loop(
                obs_path=obs_path,
                trajectories=trajectories,
                spec_findings=spec_findings,
            )

        findings = self._build_findings(
            patterns_payload,
            char_id=char_id,
            starting_index=starting_index,
            trajectories=trajectories,
        )
        return TrajectoryAnalyzerResult(
            findings=findings,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tool_calls=tool_calls,
        )

    # ---------- internals ----------

    def _run_loop(
        self,
        *,
        obs_path: Path,
        trajectories: list[Trajectory],
        spec_findings: list[Finding],
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        tool_history: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0

        base_user_text = self._build_user_text(
            obs_path=obs_path,
            trajectories=trajectories,
            spec_findings=spec_findings,
        )

        for call_idx in range(MAX_TOOL_CALLS + 1):
            must_emit = call_idx >= MAX_TOOL_CALLS
            user_text = self._compose_user_text(base_user_text, tool_history, must_emit)
            try:
                parsed, in_tok, out_tok = self._client.call(  # type: ignore[union-attr]
                    system=TRAJECTORY_ANALYZER_SYSTEM,
                    user_text=user_text + "\n\n[RECAP]\n" + TRAJECTORY_ANALYZER_RECAP,
                    response_schema=_ANALYZER_SCHEMA,
                )
            except Exception as exc:
                _log.warning("trajectory_analyzer: client error %s; emitting empty", exc)
                return [], total_in, total_out, len(tool_history)

            total_in += in_tok
            total_out += out_tok
            action = parsed.get("action", "")

            if action == "tool_call" and not must_emit:
                code = (parsed.get("code") or "").strip()
                if not code:
                    tool_history.append(
                        {"call": EXECUTE_PYTHON, "code": "", "result": "(empty code)"}
                    )
                    continue
                ep_result = execute_python(code, timeout=120)
                tool_history.append(
                    {
                        "call": EXECUTE_PYTHON,
                        "code": code[:600],  # truncated for prompt redisplay
                        "result": ep_result.to_agent_text()[:2500],
                    }
                )
                continue

            patterns = parsed.get("patterns") or []
            return list(patterns), total_in, total_out, len(tool_history)

        # Loop exited without emit (shouldn't happen — must_emit forces emit)
        return [], total_in, total_out, len(tool_history)

    def _build_user_text(
        self,
        *,
        obs_path: Path,
        trajectories: list[Trajectory],
        spec_findings: list[Finding],
    ) -> str:
        # Compact context: don't dump full trajectories (that's what
        # observations.csv is for). Just metadata so the agent knows what's
        # available.
        run_ids = sorted({t.run_id for t in trajectories if t.run_id})
        variables = sorted({t.variable for t in trajectories if t.variable})
        time_grid_hint = ""
        if trajectories and trajectories[0].time_grid:
            tg = trajectories[0].time_grid
            time_grid_hint = (
                f" Time grid (first traj): {len(tg)} points, "
                f"{tg[0]:.2f}h to {tg[-1]:.2f}h"
                if len(tg) >= 2
                else ""
            )

        spec_summaries = []
        for f in spec_findings[:20]:  # cap to keep prompt bounded
            spec_summaries.append(
                f"  - [{f.severity.value}] {f.summary[:200]} "
                f"(vars: {','.join(f.variables_involved) or '—'})"
            )

        checklist = self._build_metric_checklist(
            variables=set(variables), n_runs=len(run_ids)
        )

        return (
            f"[OBSERVATIONS_CSV]\n{obs_path}\n\n"
            f"[TRAJECTORY METADATA]\n"
            f"run_ids ({len(run_ids)}): {', '.join(run_ids[:50])}"
            f"{' ...' if len(run_ids) > 50 else ''}\n"
            f"variables ({len(variables)}): {', '.join(variables[:50])}"
            f"{' ...' if len(variables) > 50 else ''}\n"
            f"trajectories total: {len(trajectories)}.{time_grid_hint}\n\n"
            f"[CATALOG CHECKLIST FOR THIS BUNDLE]\n{checklist}\n\n"
            f"[SPEC FINDINGS — context only, do not re-emit]\n"
            + ("\n".join(spec_summaries) if spec_summaries else "  (none)")
            + "\n\n[TASK]\nWork the catalog checklist top-to-bottom. For each "
            "ready metric, EITHER call its toolkit_fn via execute_python and "
            "emit a pattern with metric_id set, OR emit a data_gap pattern "
            "naming the missing input. Silent skipping is a contract "
            "violation. After the checklist is complete you may add open-"
            "ended patterns (without metric_id) for anything the catalog "
            "doesn't cover."
        )

    @staticmethod
    def _build_metric_checklist(*, variables: set[str], n_runs: int) -> str:
        """Render a deterministic per-bundle checklist of ready catalog
        entries with applicability marked from this bundle's variables.

        The model sees this in the volatile (per-bundle) section of the
        prompt so it can't claim ignorance of which metrics apply. Each
        line is one catalog entry: id, short_description, applicability
        verdict, and the inputs the toolkit_fn needs.
        """
        # Lazy import to avoid circulars; metric_catalog imports nothing
        # from trajectory_analyzer.
        from fermdocs_characterize.agents.metric_catalog import ready_entries

        var_lower = {v.lower() for v in variables}

        def _vars_present(needed: set[str]) -> bool:
            # Loose substring match — bundle column names vary slightly
            # (dissolved_o2_mg_l vs dissolved_o2_pct, etc.).
            return all(
                any(n in v or v in n for v in var_lower) for n in needed
            )

        # Map metric_id → minimum required variable substrings + run count.
        # Cross-run metrics get min_runs > 1.
        applicability: dict[str, tuple[set[str], int, str]] = {
            "A8":  ({"biomass"}, 1, "biomass_g_l ≥ 5 points/run"),
            "A9":  ({"biomass"}, 1, "biomass_g_l ≥ 5 points/run"),
            "A10": ({"biomass"}, 1, "biomass_g_l ≥ 8 points/run"),
            "A11": ({"biomass"}, 1, "biomass_g_l ≥ 8 points/run"),
            "A14": ({"dissolved_o2"}, 1, "dissolved_o2 trajectory"),
            "A15": (set(), 1, "any controlled variable + setpoint pair"),
            "A17": ({"agitation"}, 1, "agitation_rpm + impeller_diameter_m param"),
            "A18": ({"agitation"}, 1, "agitation_rpm + reactor geometry params"),
            "A19": (set(), 2, "≥ 2 runs"),
            "A20": (set(), 3, "≥ 3 runs"),
            "A21": (set(), 5, "≥ 5 runs"),
            "B6":  ({"biomass"}, 1, "biomass + a byproduct trajectory"),
            "B10": ({"our", "cer"}, 1, "OUR + CER trajectories"),
            "B16": ({"biomass"}, 1, "substrate + biomass + (products or CO2 from CER)"),
            "C2":  ({"biomass"}, 1, "A8 result + organism in priors"),
            "C3":  ({"temperature"}, 1, "temperature_C (convert from K if needed)"),
            "C4":  ({"agitation"}, 1, "P/V from A18 + superficial gas velocity"),
            "C5":  (set(), 1, "organism in priors (observed_qs optional)"),
            "C9":  ({"our"}, 1, "OUR + kLa + C* + DO"),
            "C10": (set(), 1, "organism in priors + observed qs"),
        }

        lines = []
        for entry in ready_entries():
            mid = entry.metric_id
            if mid not in applicability:
                # Defensive: any future ready entry without an applicability
                # rule still gets surfaced so the agent can decide.
                lines.append(f"  [?] {mid}  {entry.short_description}  (run anyway, decide applicability)")
                continue
            needed_vars, min_runs, hint = applicability[mid]
            verdict = "APPLICABLE" if (
                _vars_present(needed_vars) and n_runs >= min_runs
            ) else "DATA_GAP"
            lines.append(
                f"  [{verdict}] {mid}  {entry.short_description}  ({hint})"
            )
        return "\n".join(lines)

    @staticmethod
    def _compose_user_text(
        base: str, tool_history: list[dict[str, Any]], must_emit: bool
    ) -> str:
        body = base
        if tool_history:
            body += "\n\n[TOOL HISTORY]\n"
            for entry in tool_history:
                body += f"--- {entry['call']} ---\nCODE:\n{entry['code']}\nRESULT:\n{entry['result']}\n\n"
        if must_emit:
            body += (
                "\n[FORCED]\nTool budget exhausted. You MUST emit_patterns now. "
                "If you found nothing actionable, emit an empty patterns list."
            )
        return body

    @staticmethod
    def _write_observations_csv(
        trajectories: Iterable[Trajectory], path: Path
    ) -> None:
        """Long-format flatten — mirrors fermdocs_characterize.cli._flatten_trajectories.

        Duplicated intentionally: cli.py's helper is private and the
        analyzer is in a different module. Keep the two definitions
        small and structurally identical so drift is obvious.
        """
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["run_id", "variable", "time_h", "value", "imputed", "unit"]
            )
            writer.writeheader()
            for t in trajectories:
                unit = t.unit
                for time_h, value, imputed in zip(
                    t.time_grid, t.values, t.imputation_flags, strict=False
                ):
                    writer.writerow(
                        {
                            "run_id": t.run_id,
                            "variable": t.variable,
                            "time_h": time_h,
                            "value": "" if value is None else value,
                            "imputed": imputed,
                            "unit": unit,
                        }
                    )

    def _build_findings(
        self,
        patterns: list[dict[str, Any]],
        *,
        char_id: UUID,
        starting_index: int,
        trajectories: list[Trajectory],
    ) -> list[Finding]:
        # Build a (run_id, variable) → source_observation_ids index so each
        # pattern can derive REAL ingestion observation IDs from the
        # trajectories it cites. The validator rejects findings whose
        # evidence_observation_ids don't resolve through the bundle's
        # ingestion namespace; passing run_ids there (which I did initially)
        # was the bug that produced the "cites unknown observation_id 'RUN-0001'"
        # validation failure.
        traj_obs_index: dict[tuple[str, str], list[str]] = {}
        for t in trajectories:
            traj_obs_index.setdefault((t.run_id, t.variable), []).extend(
                t.source_observation_ids
            )

        findings: list[Finding] = []
        idx = starting_index
        for p in patterns:
            try:
                f = self._coerce_pattern_to_finding(
                    p,
                    char_id=char_id,
                    idx=idx,
                    traj_obs_index=traj_obs_index,
                )
            except (ValueError, TypeError, KeyError) as exc:
                _log.info(
                    "trajectory_analyzer: dropping malformed pattern (%s): %s",
                    exc.__class__.__name__,
                    exc,
                )
                continue
            findings.append(f)
            idx += 1
        return findings

    @staticmethod
    def _coerce_pattern_to_finding(
        p: dict[str, Any],
        *,
        char_id: UUID,
        idx: int,
        traj_obs_index: dict[tuple[str, str], list[str]],
    ) -> Finding:
        pattern_kind = str(p.get("pattern_kind") or "").strip() or "unspecified"
        summary = str(p.get("summary") or "").strip()
        if not summary:
            raise ValueError("empty summary")
        if len(summary) > 500:
            summary = summary[:497] + "..."

        run_ids = [str(r) for r in (p.get("run_ids") or []) if r]
        variables = [str(v) for v in (p.get("variables_involved") or []) if v]

        time_window = None
        tw = p.get("time_window")
        if isinstance(tw, dict) and (tw.get("start") is not None or tw.get("end") is not None):
            time_window = TimeWindow(
                start=tw.get("start"),
                end=tw.get("end"),
            )

        severity_raw = (p.get("severity") or "minor").lower()
        try:
            severity = Severity(severity_raw)
        except ValueError:
            severity = Severity.MINOR

        # Catalog grounding: if metric_id matches a ready entry, this is
        # a STATISTICAL finding (verified toolkit math) and gets the higher
        # confidence cap. Unknown / pending metric_ids fall through to
        # LLM_JUDGED with the 0.85 cap.
        metric_id_raw = p.get("metric_id")
        metric_id = str(metric_id_raw).strip() if metric_id_raw else ""
        catalog_entry = CATALOG.get(metric_id) if metric_id else None
        is_statistical = bool(catalog_entry and catalog_entry.is_ready())

        cap = STATISTICAL_CONFIDENCE_CAP if is_statistical else LLM_CONFIDENCE_CAP
        confidence = float(p.get("confidence") or 0.6)
        confidence = max(0.0, min(confidence, cap))

        caveats = [str(c) for c in (p.get("caveats") or []) if c]

        statistics = dict(p.get("statistics") or {})
        # Always record pattern_kind in statistics for downstream
        # discrimination — D6b/(ii) contract.
        statistics["pattern_kind"] = pattern_kind
        if metric_id:
            statistics["metric_id"] = metric_id
        if catalog_entry is not None:
            statistics["tier"] = catalog_entry.tier

        # evidence_observation_ids must be REAL ingestion observation IDs
        # (e.g. "OBS-0042"), NOT run_ids — the validator resolves them
        # through the dossier's observation namespace and rejects unknowns.
        #
        # Derivation: union the source_observation_ids of every Trajectory
        # whose (run_id, variable) overlaps the pattern's citations. If
        # the pattern lists run_ids but no variables, take all variables
        # for those runs; if vars but no runs, take all runs for those vars.
        # This mirrors how spec findings derive their evidence_observation_ids.
        run_set = set(run_ids)
        var_set = set(variables)
        evidence_obs: list[str] = []
        seen: set[str] = set()
        for (rid, var), obs_ids in traj_obs_index.items():
            run_match = (not run_set) or (rid in run_set)
            var_match = (not var_set) or (var in var_set)
            if run_match and var_match:
                for oid in obs_ids:
                    if oid not in seen:
                        seen.add(oid)
                        evidence_obs.append(oid)
        if not evidence_obs:
            # Pattern cited runs/variables we have no trajectory for.
            # Schema requires non-empty; log and let the validator drop
            # the finding downstream so the failure is loud.
            evidence_obs = ["trajectory_pattern_unanchored"]
            _log.info(
                "trajectory_analyzer: pattern cites runs=%s vars=%s but no "
                "matching trajectory; emitting unanchored marker",
                run_ids,
                variables,
            )

        tier = (
            _TIER_FROM_CATALOG[catalog_entry.tier]
            if catalog_entry is not None
            else Tier.B  # default for un-tagged trajectory patterns
        )
        extracted_via = (
            ExtractedVia.STATISTICAL if is_statistical else ExtractedVia.LLM_JUDGED
        )

        return Finding(
            finding_id=f"{char_id}:F-{idx:04d}",
            type=FindingType.TRAJECTORY_PATTERN,
            severity=severity,
            tier=tier,
            summary=summary,
            confidence=confidence,
            extracted_via=extracted_via,
            caveats=caveats,
            competing_explanations=[],
            evidence_strength=EvidenceStrength(
                n_observations=max(len(run_ids) * len(variables), 1),
                n_independent_runs=len(run_ids),
            ),
            evidence_observation_ids=evidence_obs,
            variables_involved=variables,
            time_window=time_window,
            run_ids=run_ids,
            statistics=statistics,
        )


def build_trajectory_analyzer(
    client: GeminiCharacterizeClient | None,
) -> TrajectoryAnalyzerAgent:
    return TrajectoryAnalyzerAgent(client=client)
