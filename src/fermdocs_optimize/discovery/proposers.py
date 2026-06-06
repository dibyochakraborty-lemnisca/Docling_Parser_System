"""Proposers for equation discovery: who writes the next ODE structure.

Two implementations behind one Protocol (DIP — the loop doesn't care which):

* `TemplateProposer` — deterministic structural search over a curated family of
  kinetic forms (no LLM). Lets the whole loop run with no API key, and is the
  honest fallback when a provider isn't configured.
* `LLMSpecProposer` — the real "agent writes the equations" path: it shows the
  model its current equations and exactly where they disagree with the oracle,
  and asks for a revised ModelSpec. Gemini/Anthropic via the optimizer client.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from fermdocs_optimize.discovery.spec import DiscoveryRound, ModelSpec, ParamSpec

# Shared parameter pool (names + plausible bounds; inits are generic, NOT the
# oracle's values — the fit moves them).
_PARAMS = {
    "mu_max": ParamSpec(init=0.28, lb=0.01, ub=1.0),
    "ks": ParamSpec(init=0.5, lb=0.01, ub=50.0),
    "P_max": ParamSpec(init=101.75, lb=50.0, ub=200.0),
    "Y_inv": ParamSpec(init=1.4, lb=0.1, ub=10.0),
    "alpha": ParamSpec(init=1.75, lb=0.001, ub=5.0),
    "beta": ParamSpec(init=0.12, lb=0.001, ub=2.0),
    "km": ParamSpec(init=0.02, lb=1e-5, ub=1.0),
}


def _spec(name, mu, *, with_km, with_beta, notes) -> ModelSpec:
    names = ["mu_max", "ks", "P_max", "Y_inv", "alpha"]
    if with_beta:
        names.append("beta")
    if with_km:
        names.append("km")
    growth = "alpha*mu*X" + (" + beta*X" if with_beta else "")
    dS = f"-Y_inv*({growth})" + (" + km*M" if with_km else "") + " + D*(S_f - S)"
    dM = ("-km*M + D*(M_f - M)") if with_km else "D*(M_f - M)"
    return ModelSpec(
        name=name,
        params={n: _PARAMS[n] for n in names},
        aux={"mu": mu},
        odes={"X": "mu*X - D*X", "S": dS, "P": f"{growth} - D*P", "M": dM},
        notes=notes,
    )


# Curated structural family, simplest → richest. The loop tries them in order.
_TEMPLATES = [
    lambda: _spec("monod_linear_inhib",
                  "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)",
                  with_km=False, with_beta=False,
                  notes="Monod growth, linear product inhibition, no O2/maltose coupling."),
    lambda: _spec("monod_cubic_inhib",
                  "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)**3",
                  with_km=False, with_beta=True,
                  notes="Cubic product inhibition + non-growth-associated formation."),
    lambda: _spec("monod_o2_cubic",
                  "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)**3*O2/(K_O2+O2)",
                  with_km=False, with_beta=True,
                  notes="Add O2 limitation to growth."),
    lambda: _spec("full_o2_maltose",
                  "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)**3*O2/(K_O2+O2)",
                  with_km=True, with_beta=True,
                  notes="Full form: O2 limitation + maltose uptake feedback in dS/dM."),
    lambda: _spec("full_quadratic_inhib",
                  "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)**2*O2/(K_O2+O2)",
                  with_km=True, with_beta=True,
                  notes="Quadratic inhibition variant of the full form."),
]


class SpecProposer(Protocol):
    def propose(self, *, round_index: int, history: list[DiscoveryRound],
                data_summary: dict) -> ModelSpec | None:
        """Return the next structure to try, or None to stop."""
        ...


class TemplateProposer:
    """Deterministic walk over the structural family."""

    def propose(self, *, round_index, history, data_summary):
        if round_index >= len(_TEMPLATES):
            return None
        return _TEMPLATES[round_index]()


# --- LLM proposer -----------------------------------------------------------

_SYSTEM = """You are a bioprocess kinetic-modeling agent. Your job is to DISCOVER \
the ODE structure of a lactic-acid fed-batch fermentation by trial against a \
ground-truth oracle simulator.

You propose a model as math expressions (NOT code). State y = [X, S, P, M, O2, V] \
(biomass, substrate, product, maltose, dissolved O2, volume). Available names in \
expressions: state vars X,S,P,M,O2,V; conditions D,F,S_f,M_f,v0 (D=F/V dilution); \
fixed constants K_O2,q_O2_max,O2_sat,kLa,C_FEED; plus any parameters you declare. \
Functions allowed: Max, Min, exp, log, sqrt, Abs, Pow (use ** for powers).

Each round you see your current equations, how well they fit the DATA (per-species \
R^2), and how wrong they are on held-out conditions (peak-titer RMSE in g/L and \
R^2). When scored by cross-validation you also get a worst-fold R^2 and the spread \
across folds: a worst-fold far below the pooled R^2 means the structure generalizes \
on average but fails one operating regime — fix that regime, not just the mean. \
Revise the STRUCTURE to shrink the held-out peak RMSE: change \
rate laws, inhibition order, coupling terms, add/remove parameters. You may keep \
parameters; their values are re-fit from data automatically — do not try to guess \
the oracle's true parameter values, you cannot see them.

Return ONLY a JSON object with this shape:
{"name": str, "notes": "<your reasoning>",
 "params": {"<pname>": {"init": float, "lb": float, "ub": float}, ...},
 "aux": {"mu": "<expr>", ...},
 "odes": {"X": "<expr>", "S": "<expr>", "P": "<expr>", "M": "<expr>"}}
O2 and V ODEs default to the standard aeration/volume balance if you omit them.
"""

# A backslash that does NOT begin a valid JSON escape (" \ / b f n r t, or
# u+4hex). Gemini emits these inside strings (e.g. "\mu", "\alpha", or a "\u"
# without 4 hex digits), which makes json.loads reject the whole response.
_BAD_ESCAPE = re.compile(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})')


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, tolerating ```json fences, surrounding prose, and
    invalid backslash escapes (gemini's most common malformed-JSON failure)."""
    t = text.strip()
    if "```" in t:
        t = t.split("```", 2)[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # double any stray backslash so it becomes a literal, then retry
        return json.loads(_BAD_ESCAPE.sub(r'\\\\', t))


class LLMSpecProposer:
    """Gemini/Anthropic writes and revises the equations from oracle feedback.

    COMPOUNDING: this is a stateful, multi-turn conversation. Each round appends
    the latest oracle result as a new user turn and the model's spec as a model
    turn, so gemini sees its own full chain of attempts and reasoning — it builds
    on prior rounds instead of re-deciding from scratch. Temperature and the
    explicit "lower the RMSE" goal preserve some structural exploration so it
    doesn't anchor on one lineage."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.3):
        import os
        self._model = (model or os.environ.get("FERMDOCS_OPTIMIZE_MODEL")
                       or os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-3-pro"))
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._temperature = temperature
        self._messages: list[dict] = []  # the compounding conversation

    def _feedback_turn(self, round_index, history, data_summary) -> str:
        """The new user message for this round: only the latest result (prior
        rounds already live in the conversation)."""
        if round_index == 0 or not history:
            return (f"Data summary: {json.dumps(data_summary)}\n\n"
                    "Propose your first kinetic ODE structure as the JSON spec.")
        r = history[-1]  # the round just scored
        msg = (f"Result of your round {r.round_index} '{r.spec.name}': "
               f"CV peak_RMSE={r.oracle_peak_rmse:.2f} g/L, "
               f"CV peak_R2={r.oracle_peak_r2:.3f}")
        if r.cv_worst_fold_r2 is not None:
            # worst-fold << pooled R2 means the structure overfits one regime: a
            # richer signal than the average for the model to act on.
            msg += (f" (worst-fold R2={r.cv_worst_fold_r2:.3f}, "
                    f"spread={(r.cv_fold_r2_std or 0.0):.3f})")
        if r.cv_failed_folds:
            msg += f", {r.cv_failed_folds} fold(s) failed to fit (unstable structure)"
        msg += (f", data_P_R2={r.r2_by_species.get('P', float('nan')):.3f}"
                + ("" if r.compile_ok else f" [COMPILE ERROR: {r.error}]") + ".")
        best = min((h for h in history if h.compile_ok),
                   key=lambda h: h.oracle_peak_rmse, default=None)
        if best is not None:
            msg += f" Best so far: '{best.spec.name}' (RMSE={best.oracle_peak_rmse:.2f})."
        msg += (" Reason about WHY that helped or hurt — if the worst fold is much "
                "worse than the average, the structure overfits one operating regime; "
                "fix that, not just the mean. Then revise the STRUCTURE to lower the "
                "CV peak RMSE. Return the JSON spec.")
        return msg

    def propose(self, *, round_index, history, data_summary):
        from google import genai
        from google.genai import types

        # append this round's feedback as the next user turn (compounding)
        self._messages.append({
            "role": "user",
            "parts": [{"text": self._feedback_turn(round_index, history, data_summary)}],
        })
        client = genai.Client(api_key=self._api_key)
        resp = client.models.generate_content(
            model=self._model,
            contents=self._messages,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
                temperature=self._temperature,
            ),
        )
        if not resp.text:
            return None
        # record the model's reply so the next round builds on it
        self._messages.append({"role": "model", "parts": [{"text": resp.text}]})
        raw = _extract_json(resp.text)
        return ModelSpec(
            name=raw.get("name", f"llm_r{round_index}"),
            notes=raw.get("notes", ""),
            params={k: ParamSpec(**v) for k, v in raw["params"].items()},
            aux=raw.get("aux", {}) or {},
            odes=raw["odes"],
        )
