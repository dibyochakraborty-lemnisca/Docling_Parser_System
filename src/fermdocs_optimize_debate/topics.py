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

from fermdocs_optimize_debate.levers import DEFAULT_LEVERS, KnobLever

# Knob levers are the spine of the debate (they map to the box), so they sit above
# observed-trend topics in the ranker.
_KNOB_PRIORITY = 0.7
_TREND_PRIORITY = 0.5


def extract_opportunity_topics(
    char,
    *,
    objective_species: str = "P",
    levers: tuple[KnobLever, ...] = DEFAULT_LEVERS,
    max_trend_topics: int = 4,
) -> list[SeedTopic]:
    """Knob-anchored + trend opportunity topics, in deterministic id order."""
    findings = list(getattr(char, "findings", []) or [])
    trajectories = list(getattr(char, "trajectories", []) or [])
    topics: list[SeedTopic] = []
    counter = 0

    # 1. KNOB-ANCHORED — one per lever, grounded in its effect-variable evidence.
    for lever in levers:
        counter += 1
        cited_findings, cited_trajs = _evidence_for(findings, trajectories, lever.effect_variables)
        topics.append(SeedTopic(
            topic_id=_topic_id(counter),
            summary=lever.question.format(obj=objective_species),
            source_type=TopicSourceType.OPEN_QUESTION,
            source_id=f"lever-{lever.knob}",
            cited_finding_ids=cited_findings,
            cited_narrative_ids=[],
            cited_trajectories=cited_trajs,
            affected_variables=list(lever.effect_variables),
            severity=Severity.MAJOR,
            priority=_KNOB_PRIORITY,
        ))

    # 2. TREND — findings touching the objective or any lever's driver variables,
    #    objective-citing findings first, capped to keep the debate focused.
    driver_vars: set[str] = {objective_species}
    for lever in levers:
        driver_vars.update(lever.effect_variables)
    relevant = [f for f in findings if set(getattr(f, "variables_involved", []) or []) & driver_vars]
    relevant.sort(
        key=lambda f: objective_species in (getattr(f, "variables_involved", []) or []),
        reverse=True,
    )
    for f in relevant[:max_trend_topics]:
        counter += 1
        fvars = list(getattr(f, "variables_involved", []) or [])
        topics.append(SeedTopic(
            topic_id=_topic_id(counter),
            summary=f"Observed: {getattr(f, 'summary', '')[:160]} — is there headroom to raise peak "
                    f"{objective_species} here?",
            source_type=TopicSourceType.TREND,
            source_id=getattr(f, "finding_id", f"finding-{counter}"),
            cited_finding_ids=[getattr(f, "finding_id")] if getattr(f, "finding_id", None) else [],
            cited_narrative_ids=[],
            cited_trajectories=_trajs_for(trajectories, fvars),
            affected_variables=fvars,
            severity=Severity.MINOR,
            priority=_TREND_PRIORITY,
        ))

    return topics


def _evidence_for(findings, trajectories, variables) -> tuple[list[str], list[TrajectoryRef]]:
    vs = set(variables)
    cited_findings = [
        getattr(f, "finding_id") for f in findings
        if getattr(f, "finding_id", None) and set(getattr(f, "variables_involved", []) or []) & vs
    ]
    return cited_findings, _trajs_for(trajectories, variables)


def _trajs_for(trajectories, variables) -> list[TrajectoryRef]:
    vs = set(variables)
    return [
        TrajectoryRef(run_id=getattr(t, "run_id"), variable=getattr(t, "variable"))
        for t in trajectories
        if getattr(t, "variable", None) in vs and getattr(t, "run_id", None)
    ]


def _topic_id(n: int) -> str:
    return f"T-{n:04d}"
