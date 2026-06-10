"""Agent-decided search box: let the LLM look at the data and choose where to
search — free to extend beyond observed ranges where it judges there's headroom.

This is safe ONLY because the active-learning loop verifies every proposed
optimum on the ground-truth oracle: a too-wide box just costs a verify call, it
can never corrupt the recommendation (which is always the best oracle-verified
point). So we don't cap the agent to the data envelope or a config box — we only
enforce trivial physical floors (knobs >= 0, malt_frac in [0,1]).
"""
from __future__ import annotations

import json
import logging
import os

from fermdocs_optimize.discovery.proposers import _extract_json
from fermdocs_optimize.schema import KNOB_NAMES, Box
from fermdocs_optimize.search_space import _reconstruct_knobs

log = logging.getLogger(__name__)

# (lower floor, upper cap | None) — physical sanity only, NOT a search constraint.
_FLOORS = {"biomass": (0.0, None), "total_sub": (0.0, None),
           "malt_frac": (0.0, 1.0), "dilution": (0.0, None)}

_SYSTEM = (
    "You set the SEARCH RANGE for optimizing a lactic-acid fed-batch fermentation. "
    "You are given the operating conditions seen so far in the data (min/max for each "
    "knob) and the goal: maximize peak {obj}. Propose lower/upper search bounds for "
    "each of the four knobs: biomass, total_sub, malt_frac, dilution.\n\n"
    "You MAY extend beyond the observed ranges where the data's trend suggests there is "
    "headroom — the whole point is to beat the best condition seen, so do not just echo "
    "the observed min/max. Every point you enable will be verified on a ground-truth "
    "simulator, so reasonable extrapolation is safe and encouraged. Stay physically "
    "plausible: all knobs >= 0, malt_frac in [0,1].\n\n"
    'Return ONLY JSON: {"biomass":{"lb":<n>,"ub":<n>}, "total_sub":{"lb":<n>,"ub":<n>}, '
    '"malt_frac":{"lb":<n>,"ub":<n>}, "dilution":{"lb":<n>,"ub":<n>}, "reasoning":"<why>"}'
)


def propose_search_box(data, *, objective_species: str = "P",
                       model: str | None = None, api_key: str | None = None) -> Box | None:
    """Ask the LLM to choose the search box from the observed data. Returns None
    on any failure (caller falls back to a data-derived box)."""
    kn = _reconstruct_knobs(data)
    observed = {k: [round(float(min(kn[k])), 5), round(float(max(kn[k])), 5)]
                for k in KNOB_NAMES}
    model = (model or os.environ.get("FERMDOCS_OPTIMIZE_MODEL")
             or os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-3-pro"))
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (f"Observed operating ranges (min, max) over {data['batch'].nunique()} "
                  f"batches: {json.dumps(observed)}. Goal: maximize peak {objective_species}. "
                  "Propose the search bounds.")
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM.format(obj=objective_species),
                response_mime_type="application/json", temperature=0.2))
        raw = _extract_json(resp.text)
        # Gemini occasionally leaves stray quotes/whitespace on keys (e.g. the
        # key '"biomass"' instead of 'biomass'), which turns a normal lookup into
        # a KeyError. Normalize keys so the bounds still resolve.
        raw = {str(k).strip().strip('"').strip("'"): v for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("agent search-box call failed (%s); using data-derived box", exc)
        return None

    bounds: dict[str, tuple[float, float]] = {}
    for k in KNOB_NAMES:
        try:
            lb, ub = float(raw[k]["lb"]), float(raw[k]["ub"])
        except Exception:  # noqa: BLE001
            log.warning("agent search-box malformed for %s; using data-derived box", k)
            return None
        flo, fhi = _FLOORS[k]
        lb = max(lb, flo)
        if fhi is not None:
            ub = min(ub, fhi)
        if lb >= ub:
            log.warning("agent search-box degenerate for %s; using data-derived box", k)
            return None
        bounds[k] = (lb, ub)
    log.info("agent search box: %s | reason: %s",
             {k: tuple(round(x, 4) for x in v) for k, v in bounds.items()},
             str(raw.get("reasoning", ""))[:160])
    return Box(**bounds)
