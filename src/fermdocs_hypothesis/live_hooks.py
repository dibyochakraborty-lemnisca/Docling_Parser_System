"""LiveHooks — RunnerHooks impl backed by real LLM agents.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 2 / Stage 3.

Stage 2 ships: orchestrator + kinetics specialist + synthesizer.
Mass-transfer and metabolic specialists are stubbed (return a minimal
no-op facet) until Stage 3.

Critic and judge are also stubbed (always green-flag, judge says
"not valid" so accept-path runs). Stage 3 will wire real critic+judge.

This split keeps the Stage 2 deliverable narrow (one specialist works
end-to-end on real data) while leaving the runner contract unchanged
when Stage 3 swaps stubs for real agents.
"""

from __future__ import annotations

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.agents.orchestrator import OrchestratorAgent
from fermdocs_hypothesis.agents.specialist_kinetics import (
    SpecialistAgent,
    build_kinetics_specialist,
)
from fermdocs_hypothesis.agents.synthesizer import SynthesizerAgent, build_synthesizer
from fermdocs_hypothesis.bundle_loader import LoadedBundle
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.projector import (
    project_orchestrator,
    project_specialist,
    project_synthesizer,
)
from fermdocs_hypothesis.runner import RunnerState
from fermdocs_hypothesis.schema import (
    CritiqueFull,
    FacetFull,
    HypothesisFull,
    SpecialistRole,
)
from fermdocs_hypothesis.tools_bundle.factory import HypothesisToolBundle, make_tool_bundle


class LiveHooks:
    """Stage 2 RunnerHooks: real orchestrator + kinetics + synthesizer;
    stubbed mass_transfer + metabolic + critic + judge."""

    def __init__(
        self,
        bundle: LoadedBundle,
        *,
        client: GeminiHypothesisClient | None = None,
    ):
        self._bundle = bundle
        self._client = client or GeminiHypothesisClient()
        self._tools: HypothesisToolBundle = make_tool_bundle(bundle)
        self._orchestrator = OrchestratorAgent(self._client)
        self._kinetics = build_kinetics_specialist(self._client, self._tools)
        self._synthesizer = build_synthesizer(self._client)

    # ---- orchestrator ----

    def pick_topic(self, state: RunnerState) -> str | None:
        view = project_orchestrator(
            events=(),  # ranker doesn't need event history for topic-picking decision-quality in v0
            seed_topics=list(state.seed_topics),
            budget=state.budget,
            current_turn=state.current_turn + 1,
        )
        # Filter out used topics from view's top_topics so the LLM never
        # picks a topic the runner would reject.
        used = set(state.used_topic_ids)
        view = view.model_copy(
            update={"top_topics": [t for t in view.top_topics if t.topic_id not in used]}
        )
        if not view.top_topics:
            return None
        action, _in, _out = self._orchestrator.decide(view)
        if action.action == "select_topic":
            return action.topic_id
        return None

    # ---- specialists ----

    def contribute_facet(
        self, state: RunnerState, role: SpecialistRole, facet_id: str
    ) -> tuple[FacetFull, int, int]:
        assert state.current_topic is not None
        if role == "kinetics":
            view = project_specialist(
                events=(),
                role=role,
                current_topic=state.current_topic,
                available_findings=self._bundle.findings_pool,
                available_narratives=self._bundle.narratives_pool,
                available_trajectories=self._bundle.trajectories_pool,
                available_priors=self._bundle.priors_pool,
            )
            result = self._kinetics.contribute(view, facet_id=facet_id)
            return result.facet, result.input_tokens, result.output_tokens
        # Stage 2 stub for other specialists — minimal facet using topic citations
        return self._stub_facet(state, role, facet_id), 0, 0

    def _stub_facet(
        self, state: RunnerState, role: SpecialistRole, facet_id: str
    ) -> FacetFull:
        topic = state.current_topic
        assert topic is not None
        # Emit a minimal facet citing whatever the topic already cites, so it
        # passes citation discipline. Stage 3 replaces this with real agents.
        cited_findings = list(topic.cited_finding_ids)
        cited_narratives = list(topic.cited_narrative_ids)
        cited_trajs = list(topic.cited_trajectories)
        if not cited_findings and not cited_narratives and not cited_trajs:
            # Emergency: pull anything from the pools so validation passes
            if self._bundle.narratives_pool:
                cited_narratives = [self._bundle.narratives_pool[0].narrative_id]
            elif self._bundle.findings_pool:
                cited_findings = [self._bundle.findings_pool[0].finding_id]
        return FacetFull(
            facet_id=facet_id,
            specialist=role,
            summary=f"[stub-{role}] no real specialist in Stage 2; pass-through citation",
            cited_finding_ids=cited_findings,
            cited_narrative_ids=cited_narratives,
            cited_trajectories=cited_trajs,
            affected_variables=list(topic.affected_variables),
            confidence=0.4,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        )

    # ---- synthesizer ----

    def synthesize(
        self, state: RunnerState, hyp_id: str
    ) -> tuple[HypothesisFull, int, int]:
        assert state.current_topic is not None
        view = project_synthesizer(
            current_topic=state.current_topic,
            facets=list(state.current_facets),
        )
        result = self._synthesizer.synthesize(view, hyp_id=hyp_id)
        return result.hypothesis, result.input_tokens, result.output_tokens

    # ---- critic + judge (stub for Stage 2; Stage 3 swaps in real agents) ----

    def critique(self, state: RunnerState) -> tuple[CritiqueFull, int, int]:
        assert state.current_hypothesis is not None
        # Stage 2 stub: always green-flag (no critique authored)
        return CritiqueFull(
            hyp_id=state.current_hypothesis.hyp_id,
            flag="green",
            reasons=[],
            tool_calls_used=0,
        ), 0, 0

    def judge(self, state: RunnerState) -> tuple[bool, str, int, int]:
        # Stub: criticism not valid (because critic green-flagged)
        return False, "stub judge: critic green-flagged", 0, 0
