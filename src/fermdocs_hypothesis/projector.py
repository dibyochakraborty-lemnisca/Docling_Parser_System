"""Projector — role-shaped views from events + frozen upstream bundles.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §4.

Each view assembler takes (events, role_context) and returns a typed view.
Views are deliberately small and role-relevant; the projector enforces a
per-view truncation cap so context never blows up regardless of debate
length.

Stage 1 has no LLM, so caps are nominal; Stage 2 will tune them.
"""

from __future__ import annotations

from collections.abc import Iterable

from fermdocs_hypothesis.events import Event
from fermdocs_hypothesis.ranker import rank_topics
from fermdocs_hypothesis.schema import (
    AnalysisRef,
    BudgetSnapshot,
    CitationCatalog,
    CriticView,
    FacetFull,
    FindingRef,
    HypothesisFull,
    JudgeView,
    NarrativeRef,
    OpenQuestionRef,
    OrchestratorView,
    ResolvedPriorRef,
    SeedTopic,
    SpecialistRole,
    SpecialistView,
    SynthesizerView,
    TopicSpec,
    TrajectoryViewRef,
)
from fermdocs_hypothesis.state import (
    accepted_hypotheses,
    facets_for_current_topic,
    last_turn_outcome,
    open_questions,
    specialist_domain_tags,
)


# Per-view caps. Hard truncation; oldest first.
VIEW_CAPS = {
    "findings_per_specialist": 12,
    "narratives_per_specialist": 10,
    "trajectories_per_specialist": 8,
    "priors_per_specialist": 8,
    "questions_per_specialist": 8,
    "facets_per_topic": 6,
    "analyses_per_specialist": 6,
    "top_topics_k": 3,
    "accepted_hypotheses_in_orchestrator_view": 8,
}


def project_orchestrator(
    events: Iterable[Event],
    *,
    seed_topics: list[SeedTopic],
    budget: BudgetSnapshot,
    current_turn: int,
) -> OrchestratorView:
    events_list = list(events)
    top = rank_topics(seed_topics, events_list, k=VIEW_CAPS["top_topics_k"])
    unresolved = [q for q in open_questions(events_list) if not q.resolved]
    accepted = accepted_hypotheses(events_list)[
        -VIEW_CAPS["accepted_hypotheses_in_orchestrator_view"] :
    ]
    return OrchestratorView(
        current_turn=current_turn,
        budget_remaining=budget,
        top_topics=top,
        open_questions=unresolved,
        last_turn_outcome=last_turn_outcome(events_list),
        accepted_hypotheses_so_far=accepted,
    )


def project_specialist(
    events: Iterable[Event],
    *,
    role: SpecialistRole,
    current_topic: TopicSpec,
    available_findings: list[FindingRef],
    available_narratives: list[NarrativeRef],
    available_trajectories: list[TrajectoryViewRef],
    available_priors: list[ResolvedPriorRef],
    available_analyses: list[AnalysisRef] | None = None,
) -> SpecialistView:
    """Filter the upstream pools to what's relevant for this specialist on
    this topic.

    Filtering is intentionally generous — better to over-include than miss
    the variable that matters. Hard caps still apply at the end.
    """
    events_list = list(events)
    domain = specialist_domain_tags(role)
    topic_vars = {v.lower() for v in current_topic.affected_variables}
    cited_finding_ids = set(current_topic.cited_finding_ids)
    cited_narrative_ids = set(current_topic.cited_narrative_ids)
    cited_traj_keys = {(t.run_id, t.variable) for t in current_topic.cited_trajectories}

    def _finding_relevant(f: FindingRef) -> bool:
        if f.finding_id in cited_finding_ids:
            return True
        if any(v.lower() in topic_vars for v in f.variables_involved):
            return True
        if any(v.lower() in domain for v in f.variables_involved):
            return True
        return False

    def _narrative_relevant(n: NarrativeRef) -> bool:
        if n.narrative_id in cited_narrative_ids:
            return True
        text = (n.summary + " " + n.tag).lower()
        return any(d in text for d in domain) or any(v in text for v in topic_vars)

    def _traj_relevant(t: TrajectoryViewRef) -> bool:
        if (t.run_id, t.variable) in cited_traj_keys:
            return True
        v = t.variable.lower()
        return v in topic_vars or v in domain

    def _prior_relevant(p: ResolvedPriorRef) -> bool:
        v = p.variable.lower()
        return v in topic_vars or v in domain

    findings = [f for f in available_findings if _finding_relevant(f)]
    narratives = [n for n in available_narratives if _narrative_relevant(n)]
    trajectories = [t for t in available_trajectories if _traj_relevant(t)]
    priors = [p for p in available_priors if _prior_relevant(p)]

    # Analyses overlap when they cite a finding the topic also cites OR
    # share at least one affected_variable. Generous on purpose — better
    # to over-include a caveat than miss it.
    cited_findings_set = set(current_topic.cited_finding_ids)
    analyses: list[AnalysisRef] = []
    for a in (available_analyses or []):
        if cited_findings_set.intersection(a.cited_finding_ids):
            analyses.append(a)
            continue
        if topic_vars and any(v.lower() in topic_vars for v in a.affected_variables):
            analyses.append(a)

    # Open questions in this domain
    questions = [
        q
        for q in open_questions(events_list)
        if not q.resolved
        and (any(t.lower() in domain for t in q.tags) or any(t.lower() in topic_vars for t in q.tags))
    ]

    facets = facets_for_current_topic(events_list, current_topic.topic_id)
    # Don't show this specialist their own facet back
    facets = [f for f in facets if f.specialist != role]

    return SpecialistView(
        specialist_role=role,
        current_topic=current_topic,
        relevant_findings=findings[: VIEW_CAPS["findings_per_specialist"]],
        relevant_narratives=narratives[: VIEW_CAPS["narratives_per_specialist"]],
        relevant_trajectories=trajectories[: VIEW_CAPS["trajectories_per_specialist"]],
        relevant_priors=priors[: VIEW_CAPS["priors_per_specialist"]],
        relevant_analyses=analyses[: VIEW_CAPS["analyses_per_specialist"]],
        open_questions_in_domain=questions[: VIEW_CAPS["questions_per_specialist"]],
        prior_facets_this_topic=facets[: VIEW_CAPS["facets_per_topic"]],
    )


def project_synthesizer(
    *,
    current_topic: TopicSpec,
    facets: list[FacetFull],
) -> SynthesizerView:
    """Synthesizer sees full facets and a unioned citation catalog."""
    finding_ids: list[str] = []
    narrative_ids: list[str] = []
    trajectories = []
    seen_finding: set[str] = set()
    seen_narrative: set[str] = set()
    seen_traj: set[tuple[str, str]] = set()

    for f in facets:
        for fid in f.cited_finding_ids:
            if fid not in seen_finding:
                seen_finding.add(fid)
                finding_ids.append(fid)
        for nid in f.cited_narrative_ids:
            if nid not in seen_narrative:
                seen_narrative.add(nid)
                narrative_ids.append(nid)
        for ref in f.cited_trajectories:
            key = (ref.run_id, ref.variable)
            if key not in seen_traj:
                seen_traj.add(key)
                trajectories.append(ref)

    return SynthesizerView(
        current_topic=current_topic,
        facets=facets,
        citation_universe=CitationCatalog(
            finding_ids=finding_ids,
            narrative_ids=narrative_ids,
            trajectories=trajectories,
        ),
    )


def project_critic(
    *,
    hypothesis: HypothesisFull,
    citation_lookups: dict[str, object] | None = None,
    relevant_priors: list[ResolvedPriorRef] | None = None,
    debate_summary_one_line: str = "",
) -> CriticView:
    return CriticView(
        hypothesis=hypothesis,
        citation_lookups=dict(citation_lookups or {}),
        relevant_priors=list(relevant_priors or []),
        debate_summary_one_line=debate_summary_one_line,
    )


def project_judge(
    *,
    hypothesis: HypothesisFull,
    critique,  # CritiqueFull
    citation_lookups: dict[str, object] | None = None,
) -> JudgeView:
    return JudgeView(
        hypothesis=hypothesis,
        critique=critique,
        citation_lookups=dict(citation_lookups or {}),
    )


def filter_question_to_specialist(
    q: OpenQuestionRef, role: SpecialistRole
) -> bool:
    """Helper: True when an open question's tags overlap the specialist's
    domain. Exposed for tests so we don't depend on private filter logic.
    """
    domain = specialist_domain_tags(role)
    return any(t.lower() in domain for t in q.tags)
