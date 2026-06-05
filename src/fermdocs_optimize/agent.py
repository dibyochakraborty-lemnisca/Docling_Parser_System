"""Optimizer agent: an LLM orchestrator over the deterministic closed loop.

Where the diagnostic system asks "what went wrong," this agent asks "how do we
push the target variable as high as possible?" It interprets the objective,
chooses the model and proposer, sets the round/convergence budget, drives the
loop, and narrates the result for a bioprocess engineer.

The agent does NOT decide the numbers. `run_optimization_loop` fits the model,
proposes knobs, and simulates them on the ground-truth oracle; the achieved
titer and best operating point come from that loop, so the verdict (a confident
recommendation vs an honest "could not improve / low confidence") cannot be
fabricated by the LLM. This mirrors the recommend stage, where the rubric — not
the model — is authoritative.

Graceful degradation: with no LLM client (provider fake/none), the agent runs
the deterministic loop directly with sensible defaults. The agentic shell adds
interpretation and narration; the core works standalone.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from fermdocs_optimize import skill_loader
from fermdocs_optimize.evaluate import peak_titer_per_batch
from fermdocs_optimize.llm_clients import OptimizeLLMClient
from fermdocs_optimize.loop import refusal, run_optimization
from fermdocs_optimize.models.mechanistic import MechanisticModel
from fermdocs_optimize.proposers.optimize import OptimizeProposer
from fermdocs_optimize.schema import Box, OptimizationInput, OptimizationOutput
from fermdocs_optimize.simulators.base import Simulator
from fermdocs_optimize.tools_bundle.factory import make_optimize_tools

_log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 12
# Below this final-round model-vs-oracle agreement, the model never tracked the
# oracle — the achieved titer is still oracle-verified, but the proposed point is
# a lead, not a confident optimum. We annotate the rationale to say so.
LOW_AGREEMENT_R2 = 0.5


_PROMPT_TEMPLATE = '''You are a bioprocess optimization agent. Goal: given an experiment's seed batch
data and a feasible operating box, find the operating point (four knobs) that
maximizes the product titer P — even if the experiment is perfectly healthy.

WORKFLOW (follow in order):
1. get_experiment() — read the seed data: batches, species, the BASELINE peak
   titer you must beat, and the observed knob ranges.
2. get_box() — read the feasible search box (per-knob lb/ub). Every proposal is
   clamped to it.
2b. get_levers() — read the debated optimization levers (the prior from the
   opportunity debate: which knobs the specialists think raise titer and why).
   These are ADVISORY — the loop still searches the FULL box and the oracle
   judges everything. Use them to narrate and to reconcile your result; never to
   restrict the search.
3. get_skill("optimize-titer") and get_skill("choose-model-and-proposer") — read
   the recipe and the model/proposer guidance BEFORE running the loop.
4. run_optimization_loop(objective_species="P", model="mechanistic",
   proposer="optimize", max_rounds=6, proposals_per_round=4,
   delta_titer_threshold=2.0, oracle_search=true) — run the whole closed loop (it
   fits the model, proposes knobs, and SIMULATES them on the ground-truth oracle
   internally). Keep oracle_search=true: after the model-guided loop, it searches
   the SIMULATOR DIRECTLY across the box (dense sweep + refinement) to find the
   TRUE within-box maximum, not the surrogate's local plateau — this is the
   trustworthy maximum. Read the summary: best_achieved_titer, improvement,
   trajectory, per-round fit_target_r2 + model_vs_oracle_r2, convergence_reason,
   and oracle_search (true_box_max, n_oracle_evals, knobs_on_boundary). If
   knobs_on_boundary is non-empty, the box is the binding constraint — say so:
   the real optimum likely lies OUTSIDE those limits.
   Re-run with a different model/proposer or more rounds ONLY if unconverged or
   low-confidence (the loop is the expensive call — at most a few runs).
5. submit_optimization(payload_json="...") — a JSON string with:
     "rationale": 2-4 sentences an engineer can act on — the proposed operating
        point, achieved vs baseline titer, how convergence went, any caveat.
     "confidence_note": one honest line on how much to trust the optimum.
     "lever_reconciliation": one line explaining the oracle-verified optimum
        through the debated levers — which it CONFIRMS and which it CONTRADICTS
        (e.g. "the loop drove dilution down, confirming the kinetics lever, but
        kept malt_frac low, contradicting the metabolic lever").
   The authoritative numbers (best point, achieved titer, trajectory) are taken
   from the loop result you produced; you cannot override them here.

HONESTY: the oracle judges every proposal, so the achieved titer is real even
when the model fit is mediocre. But if model_vs_oracle_r2 stayed low after
augmentation, or the improvement is tiny, SAY SO — report the point as a lead,
not a verified optimum. Never dress up a weak result as a strong one. Always
reach submit_optimization before the step budget runs out.

=== optimizer conventions + the loop + integrity invariant (vendored) ===
__OPTIMIZE_README__
'''


def _system_prompt() -> str:
    return _PROMPT_TEMPLATE.replace("__OPTIMIZE_README__", skill_loader.load_readme())


def _load_levers(debate_output_path) -> list[dict]:
    """Read the opportunity debate's levers (advisory prior), or [] if absent.

    Lazy import keeps the optimizer free of the debate package at module load;
    the seam is just a JSON file (no engine code is pulled in)."""
    if not debate_output_path:
        return []
    try:
        from fermdocs_optimize_debate.schema import levers_from_debate_json
        return [v.model_dump() for v in levers_from_debate_json(debate_output_path)]
    except Exception:  # noqa: BLE001 — a missing/garbled debate file never blocks optimization
        _log.warning("could not load debate levers from %s", debate_output_path)
        return []


class OptimizerAgent:
    def __init__(
        self,
        client: OptimizeLLMClient | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        model: str = "gemini-3-pro",
        provider: Literal["anthropic", "gemini"] = "gemini",
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._model = model
        self._provider = provider

    def optimize(
        self,
        *,
        training_data: pd.DataFrame,
        box: Box,
        simulator: Simulator,
        baseline_titer: float | None = None,
        objective_species: str = "P",
        v0: float = 10.0,
        fallback_spec: OptimizationInput | None = None,
        debate_output_path: "str | None" = None,
        optimization_id: uuid.UUID | None = None,
        generation_timestamp: datetime | None = None,
        run_id: str | None = None,
    ) -> OptimizationOutput:
        opt_id = optimization_id or uuid.uuid4()
        ts = generation_timestamp or datetime.now(timezone.utc)
        if baseline_titer is None and objective_species in training_data.columns:
            peaks = peak_titer_per_batch(training_data, objective_species)
            baseline_titer = max(peaks.values()) if peaks else None

        levers = _load_levers(debate_output_path)
        meta = {"optimization_id": str(opt_id), "run_id": run_id,
                "generation_timestamp": ts.isoformat(), "provider": self._provider,
                "model": self._model, "grounding_levers": [v.get("lever_id") for v in levers]}

        if self._client is None:
            return self._deterministic(training_data, box, simulator, baseline_titer,
                                       objective_species, v0, meta, fallback_spec, levers)

        try:
            return self._run_loop(training_data, box, simulator, baseline_titer,
                                  objective_species, v0, meta, levers)
        except Exception as exc:  # noqa: BLE001 — the stage must never raise into the run
            _log.exception("optimization stage failed")
            out = refusal("stage_error", f"{exc}")
            out.meta = meta
            return out

    # ------------------------------------------------------------------
    def _deterministic(self, training_data, box, simulator, baseline, target, v0, meta,
                       fallback_spec=None, levers=None):
        """No LLM: run the loop with defaults (or a caller-supplied fallback_spec).
        The core works standalone."""
        if fallback_spec is not None:
            spec = fallback_spec.model_copy(update={"box": box, "objective_species": target, "v0": v0})
        else:
            # No-LLM default: fully utilise the oracle for the true box maximum.
            spec = OptimizationInput(box=box, objective_species=target, v0=v0,
                                     oracle_search=True)
        try:
            out = run_optimization(
                training_data=training_data, model=MechanisticModel(),
                proposer=OptimizeProposer(), simulator=simulator, spec=spec,
                baseline_titer=baseline)
        except Exception as exc:  # noqa: BLE001
            _log.exception("deterministic optimization failed")
            out = refusal("loop_error", f"{exc}")
        out.meta = {**meta, "model": "deterministic", "provider": "none"}
        if out.confident:
            out.selection_rationale = (
                "Ran the deterministic loop (no LLM orchestrator). "
                + out.selection_rationale + self._reconcile_suffix(out, levers)
                + self._honesty_suffix(out))
        return out

    def _run_loop(self, training_data, box, simulator, baseline, target, v0, meta, levers=None):
        tools = make_optimize_tools(training_data, box, simulator,
                                    baseline_titer=baseline, objective_species=target, v0=v0,
                                    levers=levers or [])
        dispatch = tools.dispatch()
        system = _system_prompt()
        messages = [{
            "role": "user",
            "content": ("Begin. Read the experiment and box, read the skills you need, "
                        "run the optimization loop, then submit your narration."),
        }]

        for step in range(self._max_steps):
            try:
                response = self._client.call(system, messages)
            except Exception as exc:  # noqa: BLE001
                _log.warning("optimize LLM call failed at step %d: %s", step, exc)
                # We may already have a loop result — finalize it honestly.
                if tools.state.best_result is not None:
                    return self._finalize(tools, meta)
                out = refusal("llm_error", f"{exc}")
                out.meta = meta
                return out

            action = response.get("action")
            if action == "emit":
                payload = self._parse_payload(response)
                if tools.state.best_result is None:
                    out = refusal("no_loop_run", "agent emitted before running the loop")
                    out.meta = meta
                    return out
                tools.state.narration = payload
                tools.state.submitted = True
                return self._finalize(tools, meta)

            if action == "tool_call":
                tool = response.get("tool")
                args = response.get("args") or {}
                handler = dispatch.get(tool)
                if handler is None:
                    result = {"error": f"unknown tool {tool}"}
                else:
                    try:
                        if tool == "submit_optimization":
                            result = handler(payload=self._parse_payload(response, args))
                        elif isinstance(args, dict):
                            result = handler(**args)
                        else:
                            result = handler(args)
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)}
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({"role": "user", "content": json.dumps(result)[:40000]})
                if tool == "submit_optimization" and isinstance(result, dict) and result.get("ok"):
                    return self._finalize(tools, meta)
                continue

            messages.append({"role": "assistant", "content": json.dumps(response)})
            messages.append({"role": "user", "content": "Action must be 'tool_call' or 'emit'."})

        # ran out of steps — if we have a loop result, finalize it; else refuse.
        if tools.state.best_result is not None:
            return self._finalize(tools, meta)
        out = refusal("step_budget_exhausted", "agent did not run the loop within the step budget")
        out.meta = meta
        return out

    # ------------------------------------------------------------------
    def _finalize(self, tools, meta) -> OptimizationOutput:
        """Build the final output from the authoritative loop result, grafting in
        the agent's narration. The agent cannot change the numbers."""
        out = tools.state.best_result
        assert out is not None
        out.meta = meta
        if not out.confident:
            return out  # honest refusal from the loop stands

        narration = tools.state.narration or {}
        rationale = (narration.get("rationale") or "").strip()
        note = (narration.get("confidence_note") or "").strip()
        recon = (narration.get("lever_reconciliation") or "").strip()
        base = out.selection_rationale
        if rationale:
            out.selection_rationale = f"{rationale} | basis: {base}"
        if note:
            out.selection_rationale += f" Confidence: {note[:300]}"
        if recon:
            out.selection_rationale += f" Levers: {recon[:300]}"
        out.selection_rationale += self._reconcile_suffix(out, tools.levers)
        out.selection_rationale += self._honesty_suffix(out)
        return out

    @staticmethod
    def _reconcile_suffix(out: OptimizationOutput, levers) -> str:
        """Deterministic appendix: show the debated levers (the prior) next to the
        oracle-verified optimum (the posterior), so the human sees both even if the
        LLM's prose reconciliation is thin or absent. Inform-only: we never claim a
        lever was 'right' beyond what the oracle verified."""
        levers = levers or []
        if not levers or out.best_candidate is None:
            return ""
        knobs = out.best_candidate.knobs()
        top = [v for v in levers if v.get("knobs")][:3]
        if not top:
            return ""
        parts = []
        for v in top:
            moved = ", ".join(f"{k}={knobs[k]:.3g}" for k in v["knobs"] if k in knobs)
            parts.append(f"[{v.get('lever_id', '?')}] {','.join(v['knobs'])}→{moved} (debated conf {v.get('confidence', 0):.2f})")
        return " [debated levers vs verified optimum: " + "; ".join(parts) + "]"

    @staticmethod
    def _honesty_suffix(out: OptimizationOutput) -> str:
        """Deterministic caution appended regardless of what the LLM wrote."""
        if not out.rounds:
            return ""
        last = out.rounds[-1]
        suffix = ""
        if last.model_vs_oracle_r2 < LOW_AGREEMENT_R2:
            suffix += (
                f" [caution: model-vs-oracle R²={last.model_vs_oracle_r2:.2f} in the final "
                "round — the surrogate never tracked the oracle, so treat the proposed point "
                "as an oracle-verified lead, not a confirmed global optimum.]")
        if out.improvement is not None and out.improvement <= 0:
            suffix += (
                f" [caution: no improvement over baseline ({out.improvement:+.1f} g/L) — "
                "the seed operating point may already be near-optimal in this box.]")
        return suffix

    @staticmethod
    def _parse_payload(response: dict, args: dict | None = None) -> dict:
        src = args if args is not None else response
        if isinstance(src, dict) and "payload_json" in src and src["payload_json"]:
            try:
                return json.loads(src["payload_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(src, dict) and isinstance(src.get("payload"), dict):
            return src["payload"]
        return src if isinstance(src, dict) else {}
