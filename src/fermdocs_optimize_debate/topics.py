"""Opportunity-topic extractor: characterization → list[SeedTopic].

The diagnostic stage seeds the debate from faults (DiagnosisOutput.failures). The
optimizer has no fault to find, so we seed from the controllable levers instead:

  - KNOB-ANCHORED topics (one per lever): "could moving this knob raise peak
    titer?", grounded in the characterization evidence for the knob's effect
    variables. These line up 1:1 with what the optimizer can move, so the debate
    produces a prior directly over the search box.
  - TREND topics: characterization findings that touch the objective or its
    drivers, reframed as "is there headroom here?" — surfaces observed behavior
    (substrate left over, DO dip, titer plateau) for the specialists to argue.

We reuse the engine's existing source types (no schema change): knob levers are
forward-looking OPEN_QUESTIONs; observed findings are TRENDs. `char` is duck-typed
(anything with `.findings` and `.trajectories`) so this is testable without
constructing a full CharacterizationOutput.
"""
from __future__ import annotations

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import TrajectoryRef
from fermdocs_hypothesis.schema import SeedTopic, TopicSourceType


# Topic priorities. Observed TRENDS are grounded in real measured behavior
# (substrate left over, titer plateau, DO dip), so they outrank the
# forward-looking lever questions — and a trend inherits its finding's severity,
# so a trend off a CRITICAL observation dominates. Levers are the controllable
# spine but are partly speculative ("could moving X help?"), so they sit below.
_KNOB_PRIORITY = 0.7
_TREND_PRIORITY = 0.8
# Severity ordering for picking WHICH trends to keep when capped (worst first).
_SEVERITY_RANK = {
    Severity.CRITICAL: 3, Severity.MAJOR: 2, Severity.MINOR: 1, Severity.INFO: 0,
}
def _debate_levers(levers: list) -> list:
    """The levers worth DEBATING: metadata design factors only (nitrogen source,
    feed, inoculum, ...). Derived observation-channel '.initial' levers are
    dropped here — you don't independently control 'initial acetate' or 'initial
    substrate'; the recipe (metadata) is the control. Derived levers still feed
    the optimizer surrogate; only their promotion to *debate topics* is removed.
    (Eng-review decision A2, 2026-06-13.)"""
    meta = [lv for lv in levers if getattr(lv, "source", None) == "metadata"]
    derived = [lv for lv in levers if getattr(lv, "source", None) != "metadata"]
    if derived:
        import logging
        logging.getLogger(__name__).info(
            "opportunity debate: debating %d metadata design factors; dropping "
            "%d derived channel levers from topics %s",
            len(meta), len(derived), [lv.name for lv in derived])
    return meta


def _discovered_lever_topic(lever, *, objective_species: str, counter: int,
                            findings, trajectories, effect: dict | None = None) -> SeedTopic:
    """One forward-looking opportunity topic for a metadata design factor. These
    have no single measured trajectory, so they carry their cross-run effect in
    the summary (and `effect_size` for ranking) rather than a citation — that is
    how a real lever leads the debate without a trajectory to cite."""
    if lever.kind == "numeric":
        rng = lever.observed_range
        span = f" (varied {round(rng[0], 4)}–{round(rng[1], 4)} across runs)" if rng else ""
        summary = f"Could changing {lever.name} raise peak {objective_species}?{span}"
    else:
        cats = ", ".join(lever.categories)
        summary = (f"Which {lever.name} maximizes peak {objective_species}? "
                   f"(observed: {cats})")
    effect_size = 0.0
    if effect:
        from fermdocs.analysis.cross_run import is_weak_effect
        # B2: rank by the CONDITIONED effect on the free objective (ranking_effect)
        # when present — confounded/underpowered levers carry ~0 and sink. Falls
        # back to the unconditioned norm_effect pre-B1' (no target stratum).
        effect_size = float(effect.get("ranking_effect", effect.get("norm_effect") or 0.0))
        delta, n = effect.get("delta"), effect.get("n")
        best = effect.get("best_setting")
        if delta is not None:
            if effect.get("confounded"):
                # Effect not attributable -> must not lead the debate as a lever.
                effect_size = 0.0
                qualifier = (f" (CONFOUNDED: {effect.get('confounded_with')}; the effect "
                             "is not separable / not attributable — do NOT argue this as a "
                             "causal lever)")
            elif is_weak_effect(effect.get("norm_effect")):
                qualifier = " (WEAK: small vs run-to-run scatter; a lead to test, not validated)"
            else:
                qualifier = ""
            summary += (f" Across {n} runs this associates with {delta:+g} "
                        f"{objective_species} (best observed: {best}); observational, "
                        f"not proven causal.{qualifier}")
    return SeedTopic(
        topic_id=_topic_id(counter),
        summary=summary,
        source_type=TopicSourceType.OPEN_QUESTION,
        source_id=f"lever-{lever.name}",
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        affected_variables=[],
        severity=Severity.MAJOR,
        priority=_KNOB_PRIORITY,
        effect_size=min(max(effect_size, 0.0), 1.0),
    )


def extract_opportunity_topics(
    char,
    *,
    objective_species: str = "product_g_l",
    discovered_levers: list | None = None,
    lever_effects: dict[str, dict] | None = None,
    max_trend_topics: int = 8,
) -> list[SeedTopic]:
    """Knob-anchored + trend opportunity topics, in deterministic id order.

    Knob-anchored topics come from ``discovered_levers`` — the experiment's OWN
    levers, found from run_conditions metadata + varying observation channels —
    so the debate argues the real levers (nitrogen source, feed conc, ...). There
    is NO fixed LABS knob fallback (de-LABS, 2026-06-16): when no levers are
    discovered, the trends carry the debate."""
    findings = list(getattr(char, "findings", []) or [])
    trajectories = list(getattr(char, "trajectories", []) or [])
    topics: list[SeedTopic] = []
    counter = 0

    # 1. KNOB-ANCHORED — one per metadata design factor, ranked by cross-run
    #    effect size. Derived channel levers are NOT debated (see _debate_levers).
    #    No discovered levers -> no knob topics; the trends below carry the debate.
    effects = lever_effects or {}
    for lever in _debate_levers(discovered_levers or []):
        counter += 1
        topics.append(_discovered_lever_topic(
            lever, objective_species=objective_species, counter=counter,
            findings=findings, trajectories=trajectories,
            effect=effects.get(lever.name)))
    driver_vars: set[str] = {objective_species}

    # 2. TREND — findings touching the objective (or driver variables in the
    #    fallback path). Keep the most important first: objective-citing AND
    #    higher-severity observations lead, so the cap keeps the trends that
    #    matter most. Each topic inherits its finding's real severity.
    relevant = [f for f in findings if set(getattr(f, "variables_involved", []) or []) & driver_vars]
    relevant.sort(
        key=lambda f: (
            objective_species in (getattr(f, "variables_involved", []) or []),
            _SEVERITY_RANK.get(getattr(f, "severity", Severity.MAJOR), 2),
        ),
        reverse=True,
    )
    for f in relevant[:max_trend_topics]:
        counter += 1
        fvars = list(getattr(f, "variables_involved", []) or [])
        cf = [getattr(f, "finding_id")] if getattr(f, "finding_id", None) else []
        ct = _trajs_for(trajectories, fvars)
        topics.append(SeedTopic(
            topic_id=_topic_id(counter),
            summary=f"Observed: {getattr(f, 'summary', '')[:160]} — is there headroom to raise peak "
                    f"{objective_species} here?",
            source_type=TopicSourceType.TREND,
            source_id=getattr(f, "finding_id", f"finding-{counter}"),
            cited_finding_ids=cf,
            cited_narrative_ids=[],
            cited_trajectories=ct,
            affected_variables=fvars,
            # Inherit the finding's severity (was hardcoded MINOR) so a trend off
            # a CRITICAL observation ranks at the top, not the bottom.
            severity=getattr(f, "severity", Severity.MAJOR),
            priority=_TREND_PRIORITY,
        ))

    return topics


def _trajs_for(trajectories, variables) -> list[TrajectoryRef]:
    vs = set(variables)
    return [
        TrajectoryRef(run_id=getattr(t, "run_id"), variable=getattr(t, "variable"))
        for t in trajectories
        if getattr(t, "variable", None) in vs and getattr(t, "run_id", None)
    ]


def _topic_id(n: int) -> str:
    return f"T-{n:04d}"
