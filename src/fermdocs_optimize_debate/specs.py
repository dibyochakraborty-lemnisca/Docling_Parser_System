"""Optimization specialist specs.

Same engine, same domain expertise as the diagnostic specialists, but the goal is
flipped: instead of explaining a fault, each specialist hunts for HEADROOM to
raise peak product titer and names the controllable lever that would capture it —
or says honestly that its domain sees no headroom.

The optimization framing (task / invariants / recap) is authored ONCE here and
shared across the three personas; only the domain identity differs per specialist.
The synthesizer/critic/judge are reused unchanged (they are evidence-quality
machinery): the synthesizer already asks for a "concrete next-batch parameter
change," which is exactly an optimization recommendation.
"""
from __future__ import annotations

from typing import Any

from fermdocs_hypothesis.agents.specialist_base import (
    SpecialistAgent,
    make_user_question_invariant,
)
from fermdocs_hypothesis.prompts import ToolHint
from fermdocs_hypothesis.tools_bundle.factory import (
    EXECUTE_PYTHON,
    GET_NARRATIVE_OBSERVATIONS,
    GET_PRIORS,
    HypothesisToolBundle,
    QUERY_BUNDLE,
    QUERY_RELATIONSHIP,
)

# The lever vocabulary the specialists must speak so a facet maps to a knob.
_KNOB_LINE = (
    "The four controllable levers are: initial biomass (X0), total initial"
    " substrate (S+M), the glucose/maltose split (malt_frac), and the"
    " feed/dilution rate. Frame your lever as one of these."
)

_OPT_MISSION = (
    "You are part of an OPTIMIZATION debate, NOT a fault hunt. The experiment may"
    " be perfectly healthy. Your job is to find where there is HEADROOM to raise"
    " the peak product titer from your domain's mechanism, and which controllable"
    f" lever would capture it. {_KNOB_LINE} If your domain sees no headroom for the"
    " current topic, say so explicitly and defer to your peers — do not invent a"
    " lever. You remain observational: cite evidence; propose a DIRECTION and a"
    " mechanism, never a magnitude you cannot ground (the simulator oracle will"
    " verify the actual gain downstream)."
)

_OPT_TASK_SPEC = """\
Read the view, optionally call tools to fetch more data, then contribute ONE
facet on the current_topic as an OPTIMIZATION LEVER from your domain angle:
  - what to change (which lever) and in which direction,
  - the mechanism by which it should raise peak titer,
  - the measured variables it acts on (put them in affected_variables: X, S, M,
    P, V — this is what maps your lever onto a controllable knob).
If your domain sees no headroom here, say so in the summary and defer.

Tool budget: up to 12 tool calls before you must contribute_facet.

Citation policy:
  - cited_finding_ids for range/cohort findings; cited_narrative_ids for prose
    evidence; cited_trajectories when you read a curve in your view.
"""

_OPT_RECAP = """\
Output one JSON action.

When tool_call: {"action":"tool_call","tool":"<name>","args":{...}}
When done: {"action":"contribute_facet","summary":..., "cited_finding_ids":[...],
"cited_narrative_ids":[...], "cited_trajectories":[{"run_id":..., "variable":...}],
"affected_variables":[...], "confidence":<0..0.85>, "confidence_basis":"schema_only"|"process_priors"|"cross_run"}

Hard rules:
  - Frame the facet as an optimization lever (or an explicit "no headroom / defer").
  - Stay in your domain.
  - At least one citation field must be non-empty.
  - Name the lever's measured variables in affected_variables so it maps to a knob.
"""

_TOOL_HINTS = (
    ToolHint(name=QUERY_BUNDLE,
             purpose="search findings/narratives/trajectories by id or substring (scope: 'finding'|'narrative'|'trajectory')"),
    ToolHint(name=GET_PRIORS,
             purpose="organism-aware variable bounds (range, typical, source)"),
    ToolHint(name=GET_NARRATIVE_OBSERVATIONS,
             purpose="filter narrative_observations by run_id/tag/variable"),
    ToolHint(name=QUERY_RELATIONSHIP,
             purpose="test a design factor's cross-run effect on titer (args: lever); "
                     "returns delta/direction/n/best_setting. Use to CHECK a lever, "
                     "then cite its assoc_id (WRA-*) in cited_association_ids"),
    ToolHint(name=EXECUTE_PYTHON,
             purpose="run short pandas analysis over the data (args: code); `obs` "
                     "(observations.csv: run_id,variable,time_h,value) is pre-loaded. "
                     "Use to verify a claim against the curves instead of asserting"),
    ToolHint(name="contribute_facet", purpose="TERMINAL: emit your lever facet and exit"),
)

# Per-specialist domain identity (the expertise is unchanged from diagnosis; only
# the mission above is new).
_KINETICS_DOMAIN = """\
Your domain is growth and consumption rates: μ (specific growth rate), q_s
(substrate uptake), q_p (product formation), yield coefficients (Y_x/s, Y_p/s),
growth-phase transitions, substrate-limitation (Monod) and substrate-inhibition
(Haldane) kinetics. Look for kinetic headroom: substrate below saturation,
unspent yield, a growth phase that could be pushed. You do NOT opine on DO/kLa
(mass_transfer) or pathway metabolism (metabolic)."""

_MASS_TRANSFER_DOMAIN = """\
Your domain: dissolved oxygen (DO), kLa, OUR, CER, RQ, agitation/mixing, gas
hold-up, foam, pH/temperature excursions, vessel volume. Look for transport
headroom: an O2-transfer ceiling that caps achievable biomass/feed, a feed/
dilution rate the vessel could support better. You do NOT opine on intrinsic
growth/uptake kinetics (kinetics) or pathway metabolism (metabolic)."""

_METABOLIC_DOMAIN = """\
Your domain: pathway implications, byproduct accumulation (acetate, lactate,
formate, etc.), induction stress, organism quirks (glucose repression, overflow
metabolism), product inhibition, nutrient limits beyond carbon, the
glucose/maltose split's effect on flux toward product. Look for metabolic
headroom: a substrate split or feed strategy that shifts flux to product, relief
of product inhibition. You do NOT opine on intrinsic kinetics (kinetics) or DO/
kLa physics (mass_transfer)."""


def _opt_identity(domain: str) -> str:
    return f"{_OPT_MISSION}\n\n{domain}"


def _shared_invariants(role: str) -> tuple[str, ...]:
    return (
        "Frame every facet as an optimization lever (what to change, direction,"
        " mechanism) OR an explicit 'no headroom from my domain; deferring'.",
        "Name the lever's measured variables in affected_variables (X/S/M/P/V) so"
        " it maps to a controllable knob (biomass/total_sub/malt_frac/dilution).",
        f"Stay in the {role} domain; defer other levers to peers.",
        "Every facet must cite ≥1 finding, narrative, or trajectory.",
        "When relevant_within_run associations are present, GROUND your lever"
        " claim in them: put the relevant assoc_id(s) (WRA-*) in"
        " cited_association_ids, state the magnitude + direction and that it is an"
        " observational association across N runs (not proven causal), and set"
        " confidence_basis='cross_run'. A design factor that measurably moved the"
        " objective beats a speculative one. (cited_association_ids counts as"
        " grounding, so a design-factor facet is not flagged schema_only.)",
        "Confidence ≤ 0.85; thin evidence → low confidence and say so.",
        "Propose direction + mechanism, not a magnitude — the simulator oracle"
        " verifies the actual titer gain downstream; do NOT overclaim the number.",
        "If you used a process_priors lookup, set confidence_basis='process_priors'.",
        make_user_question_invariant(role),
    )


def _spec(role: str, domain: str) -> dict[str, Any]:
    return {
        "role": role,
        "system_identity": _opt_identity(domain),
        "invariants": _shared_invariants(role),
        "task_spec": _OPT_TASK_SPEC,
        "tool_hints": _TOOL_HINTS,
        "recap": _OPT_RECAP,
    }


OPT_KINETICS_SPEC = _spec("kinetics", _KINETICS_DOMAIN)
OPT_MASS_TRANSFER_SPEC = _spec("mass_transfer", _MASS_TRANSFER_DOMAIN)
OPT_METABOLIC_SPEC = _spec("metabolic", _METABOLIC_DOMAIN)


def build_opt_kinetics_specialist(client, tools: HypothesisToolBundle) -> SpecialistAgent:
    return SpecialistAgent(client=client, spec=OPT_KINETICS_SPEC, tools=tools, role="kinetics")


def build_opt_mass_transfer_specialist(client, tools: HypothesisToolBundle) -> SpecialistAgent:
    return SpecialistAgent(client=client, spec=OPT_MASS_TRANSFER_SPEC, tools=tools, role="mass_transfer")


def build_opt_metabolic_specialist(client, tools: HypothesisToolBundle) -> SpecialistAgent:
    return SpecialistAgent(client=client, spec=OPT_METABOLIC_SPEC, tools=tools, role="metabolic")
