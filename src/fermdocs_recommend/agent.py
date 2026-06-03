"""Recommendation agent: skills-guided brewtwin bake-off.

The agent reads the vendored brewtwin skills, fits all three model families in
the sandbox (computing a build_report per family), and proposes counterfactual
interventions. It does NOT decide the winner — the deterministic rubric
(rubric.py) judges the candidate reports the agent produced, so the verdict
(confident recommendation vs honest refusal) cannot be fabricated by the LLM.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fermdocs.bundle import BundleReader

from fermdocs_recommend import rubric, skill_loader
from fermdocs_recommend.llm_clients import RecommendLLMClient
from fermdocs_recommend.schema import (
    CandidateReport,
    Intervention,
    RecommendationMeta,
    RecommendationOutput,
)
from fermdocs_recommend.tools_bundle.factory import make_recommend_tools

_log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 20
_MODEL_TYPES = ("mechanistic", "surrogate", "hybrid")


_PROMPT_TEMPLATE = '''You are a bioprocess modeling agent. Goal: given confirmed hypotheses about a
fermentation run, fit brewtwin models to the run's measured trajectories, pick the
best-supported family, and propose a quantitative, simulation-backed recommendation —
or honestly refuse when no model is trustworthy.

WORKFLOW (follow in order):
1. get_hypotheses() — read the confirmed hypotheses; note affected_variables + the
   qualitative actionable_recommendation each carries.
2. get_data_feed() — see run_ids, the state/control split, the feed_var, per-variable
   point counts, the observations_csv_path, and the leave_one_run_out split.
3. get_skill("fit-mechanistic-model" | "fit-surrogate-model" | "fit-hybrid-model" |
   "analyze-and-interpret") — read the recipes BEFORE writing code. They carry the
   correct brewtwin API and the gotchas. The shared conventions are below.
4. execute_python(code) — WRITE AND RUN the fits yourself, guided by the skills. This is the
   point of the agent: you make the modelling decisions (families, channels, structure,
   hyperparameters) and author the brewtwin code; the helpers only remove brittle plumbing.
   For EACH family you attempt, run a SMALL HYPERPARAMETER SEARCH inside a single
   execute_python call (do NOT one-shot — one-shot gives poor or diverged fits even when a good
   one exists):
     * mechanistic: a few initial guesses for mu_max/Ks (multi-start) x 1-2 learning rates
     * surrogate (Neural ODE / LSTM): a few widths/depths x learning rates (x epochs)
     * hybrid: a couple of residual-MLP sizes / regularisation weights
   For each config: build the model with the REAL brewtwin API exactly as the skill shows, fit
   it, simulate on the held-out (leave-one-run-out) run, and score with
   fermdocs_recommend.brewtwin_metrics.build_report on the held-out REAL points. Keep the BEST
   config per family (highest worst-channel held-out R^2); wrap each config in try/except so one
   failure does not lose the rest. Importable: brewtwin (use it as the skills show),
   fermdocs_recommend.data_feed (build_feed, get_real_observations, leave_one_run_out,
   detect_feed_var), fermdocs_recommend.brewtwin_metrics (build_report, fit_metrics). float64 is
   auto-enabled. Print ONE JSON object mapping each family to
   {"model_type","attempted","disqualified","disqualification_reason","report": <best
   build_report>, "hyperparameters": <winning config>}.
   (A convenience baseline fermdocs_recommend.fit_kit.run_bakeoff(path, biomass=..., substrate=...,
   feed_var=..., volume_var=..., n_adam=..., n_epochs=...) exists; you MAY call it per-config with
   different settings to drive your search, but you are expected to TUNE, not one-shot.)
5. After fitting, optionally simulate 1-3 counterfactual interventions on the BEST family
   (mutate species.conc for an initial-condition change, or pass controls= to JaxSolver.solve /
   a control-augmented surrogate for a knob change) and record predicted vs baseline for the
   objective (default: maximize product/penicillin final value; fall back to biomass when
   product is absent).
6. submit_recommendation(payload_json="...") — a JSON string with keys:
     "candidates": a list of three objects, each
        {"model_type": "mechanistic"|"surrogate"|"hybrid", "attempted": true|false,
         "disqualified": true|false, "disqualification_reason": str|null,
         "report": <the build_report dict, or null>}
     "interventions": list of {"description", "knob", "objective_metric",
         "baseline_value", "predicted_value", "delta", "in_coverage", "rationale"}
     "grounding_hyp_ids": ["H-0001", ...]
   DO NOT put recommended_model/confident in the payload — the system runs the rubric on
   your candidate reports to decide the winner or refusal. Your job is to produce honest,
   well-scored fits, not to pick the winner.

If brewtwin is not importable, or the data is too sparse to fit anything, submit with
candidates marked attempted:false/disqualified and an empty interventions list — the
rubric will emit the correct refusal.

STEP DISCIPLINE (important): get_hypotheses once, get_data_feed once, read the skills you need,
then run your whole hyperparameter search INSIDE one (at most two) execute_python call(s) — put
the per-family config loop in the sandbox code, not across agent steps (each step re-imports JAX
~9s). If a call errors, read stderr, fix, retry once, then submit with whatever you have (the
rubric handles disqualified families). Always reach submit_recommendation before the budget runs
out; a run that never submits is wasted.

=== GUIDANCE ===
- You drive the modelling: read the skills, choose families + channels + hyperparameters, write
  the brewtwin code, tune it. The helpers (data_feed, brewtwin_metrics, optional fit_kit) only
  remove brittle plumbing (loading, scoring) — the model choices are yours.
- TUNE, don't one-shot: a single fixed initial guess or learning rate often yields a poor or
  diverged fit even when a good one exists (multi-start fixes bad mechanistic guesses; a lower
  learning rate fixes surrogate divergence). Always compare configs on HELD-OUT R^2, keep the best.
- CHANNEL SELECTION (this decides whether you get a usable recommendation): model ONLY the
  channels that are (a) relevant to the hypothesis's affected_variables and (b) trackable. Scoring
  is on the WORST channel, so one channel a simple model cannot represent — e.g. a feed-driven
  sawtooth substrate under pulsed feeding — will sink the candidate even if the others fit well.
  Prefer a SMALL, decision-relevant set (often biomass alone, or biomass + product).
- After fitting, you may run another execute_python to simulate an intervention on the best model
  and report predicted vs baseline for the objective; if unsure, submit an empty interventions list.

=== brewtwin shared conventions + model-selection hierarchy (vendored) ===
__BREWTWIN_README__
'''


def _system_prompt() -> str:
    return _PROMPT_TEMPLATE.replace("__BREWTWIN_README__", skill_loader.load_readme())


class RecommendationAgent:
    def __init__(
        self,
        client: RecommendLLMClient | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        model: str = "gemini-3-pro",
        provider: Literal["anthropic", "gemini"] = "gemini",
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._model = model
        self._provider = provider

    def recommend(
        self,
        bundle: BundleReader,
        hypothesis_output_path: Path | None = None,
        *,
        recommendation_id: uuid.UUID | None = None,
        generation_timestamp: datetime | None = None,
        run_id: str | None = None,
    ) -> RecommendationOutput:
        rec_id = recommendation_id or uuid.uuid4()
        ts = generation_timestamp or datetime.now(timezone.utc)

        # Routing: data-rich families with a cached base model (IndPenSim) use the
        # pre-trained path (no on-the-fly fitting, no LLM needed). Everything else
        # falls through to the on-the-fly agent loop.
        try:
            from fermdocs_recommend import registry

            family = registry.process_family_of(bundle.dir)
            if registry.has_base_model(family):
                return self._cached_recommend(bundle, hypothesis_output_path, rec_id, ts, run_id, family)
        except Exception:  # noqa: BLE001 — never let routing break the stage
            _log.exception("cached-model routing failed; falling back to on-the-fly")

        if self._client is None:
            return self._error_output(rec_id, ts, run_id, error="no_llm_client")

        try:
            return self._run_loop(bundle, hypothesis_output_path, rec_id, ts, run_id)
        except Exception as exc:  # noqa: BLE001 — the stage must never raise into the run
            _log.exception("recommendation stage failed")
            return self._error_output(rec_id, ts, run_id, error=f"stage_error:{exc}")

    def _cached_recommend(self, bundle, hyp_path, rec_id, ts, run_id, family) -> RecommendationOutput:
        """Pre-trained path: load the cached base model, score it on the run, judge."""
        from fermdocs_recommend import registry

        # Read the confirmed hypotheses (id + affected_variables) so interventions
        # can be proposed against the diagnosed mechanism, not a blind list.
        hyps = []
        if hyp_path and Path(hyp_path).exists():
            try:
                data = json.loads(Path(hyp_path).read_text())
                hyps = [{"hyp_id": h.get("hyp_id"), "affected_variables": h.get("affected_variables", [])}
                        for h in data.get("final_hypotheses", []) if h.get("hyp_id")]
            except Exception:  # noqa: BLE001
                pass
        grounding = [h["hyp_id"] for h in hyps]

        try:
            cand = registry.cached_candidate(bundle.dir, family, hypotheses=hyps)
        except Exception as exc:  # noqa: BLE001
            _log.exception("cached candidate failed")
            return self._error_output(rec_id, ts, run_id, error=f"cached_model_error:{exc}")

        payload = {"candidates": [cand], "interventions": cand.get("interventions", []),
                   "grounding_hyp_ids": grounding}
        out = self._build_output(payload, rec_id, ts, run_id)
        out.meta.model = f"cached:{family}"
        # Surface the best actionable intervention in the rationale.
        best = max((i for i in out.interventions if i.delta is not None),
                   key=lambda i: i.delta, default=None)
        if best is not None and best.delta and best.delta > 0:
            out.selection_rationale += (
                f" Recommended action: {best.description} → predicted peak titer "
                f"{best.predicted_value} g/L vs {best.baseline_value} baseline "
                f"(+{best.delta}){'' if best.in_coverage else '; extrapolation — validate experimentally'}."
            )
        return out

    # ------------------------------------------------------------------
    def _run_loop(self, bundle, hyp_path, rec_id, ts, run_id) -> RecommendationOutput:
        tools = make_recommend_tools(bundle, hypothesis_output_path=hyp_path)
        dispatch = tools.dispatch()
        system = _system_prompt()
        messages = [
            {
                "role": "user",
                "content": (
                    "Begin. Read the hypotheses and data, read the skills you need, fit "
                    "all three families in the sandbox, then submit your candidate reports."
                ),
            }
        ]
        emit_payload: dict | None = None

        for step in range(self._max_steps):
            try:
                response = self._client.call(system, messages)
            except Exception as exc:  # noqa: BLE001
                _log.warning("recommend LLM call failed at step %d: %s", step, exc)
                return self._error_output(rec_id, ts, run_id, error=f"llm_error:{exc}")

            action = response.get("action")

            if action == "emit":
                emit_payload = self._parse_payload(response)
                break

            if action == "tool_call":
                tool = response.get("tool")
                args = response.get("args") or {}
                handler = dispatch.get(tool)
                if handler is None:
                    result = {"error": f"unknown tool {tool}"}
                else:
                    try:
                        if tool == "submit_recommendation":
                            payload = self._parse_payload(response, args)
                            result = handler(payload=payload)
                        elif isinstance(args, dict):
                            result = handler(**args)
                        else:
                            result = handler(args)
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)}
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({"role": "user", "content": json.dumps(result)[:60000]})
                if tool == "submit_recommendation" and isinstance(result, dict) and result.get("ok"):
                    emit_payload = tools.state.recommendation_payload
                    break
                continue

            messages.append({"role": "assistant", "content": json.dumps(response)})
            messages.append({"role": "user", "content": "Action must be 'tool_call' or 'emit'."})

        if emit_payload is None:
            # ran out of steps without submitting — honest refusal, never an error
            return self._refusal_output(
                rec_id, ts, run_id,
                reason=rubric.REFUSAL_BUDGET,
                rationale="agent did not submit within the step budget",
            )
        return self._build_output(emit_payload, rec_id, ts, run_id)

    @staticmethod
    def _parse_payload(response: dict, args: dict | None = None) -> dict:
        """Extract the recommendation payload from an emit/submit response."""
        src = args if args is not None else response
        if isinstance(src, dict) and "payload_json" in src:
            try:
                return json.loads(src["payload_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(src, dict) and "payload" in src and isinstance(src["payload"], dict):
            return src["payload"]
        return src if isinstance(src, dict) else {}

    # ------------------------------------------------------------------
    def _build_output(self, payload: dict, rec_id, ts, run_id) -> RecommendationOutput:
        raw_candidates = payload.get("candidates", []) or []
        normalized = self._normalize_candidates(raw_candidates)

        # The rubric is authoritative for the verdict.
        verdict = rubric.select(normalized, objective_in_coverage=True)
        scored = verdict["scored"]

        candidates: list[CandidateReport] = []
        for c in normalized:
            mt = c["model_type"]
            s = scored.get(mt, {})
            report = c.get("report") or {}
            candidates.append(
                CandidateReport(
                    model_type=mt,
                    attempted=bool(c.get("attempted")),
                    disqualified=bool(c.get("disqualified")),
                    disqualification_reason=c.get("disqualification_reason"),
                    selection_r2=s.get("selection_r2"),
                    selection_rmse=s.get("selection_rmse"),
                    good_fit=s.get("good_fit"),
                    good_fit_reason=s.get("good_fit_reason"),
                    plausible=s.get("plausible"),
                    offending_params=s.get("offending_params"),
                    stalled=s.get("stalled"),
                    eligible_species=s.get("eligible_species"),
                    fitted_parameters=(report.get("fitted_parameters") if isinstance(report, dict) else None),
                    report=report or None,
                )
            )

        recommended = verdict["recommended_model"]
        interventions: list[Intervention] = []
        if recommended != "none":
            interventions = self._coerce_interventions(payload.get("interventions", []))

        rationale = verdict["selection_rationale"]
        llm_note = (payload.get("selection_rationale") or "").strip()
        if llm_note:
            rationale = f"{rationale} | agent note: {llm_note[:400]}"

        return RecommendationOutput(
            meta=self._meta(rec_id, ts, run_id),
            recommended_model=recommended,
            confident=verdict["confident"],
            refusal_reason=verdict["refusal_reason"],
            selection_rationale=rationale,
            candidates=candidates,
            interventions=interventions,
            grounding_hyp_ids=[str(h) for h in payload.get("grounding_hyp_ids", []) if h],
        )

    @staticmethod
    def _normalize_candidates(raw: list) -> list[dict]:
        by_type: dict[str, dict] = {}
        for c in raw:
            if not isinstance(c, dict):
                continue
            mt = c.get("model_type")
            if mt not in _MODEL_TYPES:
                continue
            by_type[mt] = {
                "model_type": mt,
                "attempted": bool(c.get("attempted", c.get("report") is not None)),
                "disqualified": bool(c.get("disqualified")),
                "disqualification_reason": c.get("disqualification_reason") or c.get("reason"),
                "report": c.get("report") if isinstance(c.get("report"), dict) else None,
            }
        # ensure all three families are represented (missing => not attempted)
        for mt in _MODEL_TYPES:
            by_type.setdefault(
                mt, {"model_type": mt, "attempted": False, "disqualified": False,
                     "disqualification_reason": "not attempted", "report": None}
            )
        return [by_type[mt] for mt in _MODEL_TYPES]

    @staticmethod
    def _coerce_interventions(raw: list) -> list[Intervention]:
        out: list[Intervention] = []
        for i, item in enumerate(raw or []):
            if not isinstance(item, dict):
                continue
            try:
                if not item.get("intervention_id"):
                    item = {**item, "intervention_id": f"R-I-{i + 1:04d}"}
                if not item.get("description"):
                    item = {**item, "description": item.get("knob") or "intervention"}
                out.append(Intervention(**{k: v for k, v in item.items() if k in Intervention.model_fields}))
            except Exception:  # noqa: BLE001
                continue
        return out

    def _meta(self, rec_id, ts, run_id, error: str | None = None) -> RecommendationMeta:
        return RecommendationMeta(
            recommendation_id=rec_id,
            run_id=run_id,
            generation_timestamp=ts,
            model=self._model,
            provider=self._provider,
            error=error,
        )

    def _refusal_output(self, rec_id, ts, run_id, *, reason: str, rationale: str) -> RecommendationOutput:
        return RecommendationOutput(
            meta=self._meta(rec_id, ts, run_id),
            recommended_model="none",
            confident=False,
            refusal_reason=reason,
            selection_rationale=rationale,
            candidates=[],
            interventions=[],
            grounding_hyp_ids=[],
        )

    def _error_output(self, rec_id, ts, run_id, *, error: str) -> RecommendationOutput:
        return RecommendationOutput(
            meta=self._meta(rec_id, ts, run_id, error=error),
            recommended_model="none",
            confident=False,
            refusal_reason=rubric.REFUSAL_STAGE_ERROR,
            selection_rationale=error,
            candidates=[],
            interventions=[],
            grounding_hyp_ids=[],
        )
