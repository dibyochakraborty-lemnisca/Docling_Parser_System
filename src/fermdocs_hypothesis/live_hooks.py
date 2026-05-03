"""LiveHooks — RunnerHooks impl backed by real LLM agents.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 3.

Stage 3 ships ALL agents real:
  - orchestrator (deterministic ranker + LLM topic-pick over top-K)
  - specialist_kinetics, specialist_mass_transfer, specialist_metabolic
  - synthesizer (single-call merge)
  - critic (ReAct loop with execute_python; max 6 tool calls)
  - judge (single-call structured output; no debate history)

The runner contract (RunnerHooks Protocol) is unchanged from Stage 1
stubs and Stage 2 partial-LLM hooks. Forward-compat seams (LangGraph,
HITL, SuperMemory) are still intact.
"""

from __future__ import annotations

from fermdocs_hypothesis.agents.critic import CriticAgent, build_critic
from fermdocs_hypothesis.agents.judge import JudgeAgent, build_judge
from fermdocs_hypothesis.agents.orchestrator import OrchestratorAgent
from fermdocs_hypothesis.agents.specialist_base import SpecialistAgent
from fermdocs_hypothesis.agents.specialist_kinetics import build_kinetics_specialist
from fermdocs_hypothesis.agents.specialist_mass_transfer import (
    build_mass_transfer_specialist,
)
from fermdocs_hypothesis.agents.specialist_metabolic import build_metabolic_specialist
from fermdocs_hypothesis.agents.synthesizer import SynthesizerAgent, build_synthesizer
from fermdocs_hypothesis.bundle_loader import LoadedBundle
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.projector import (
    project_critic,
    project_judge,
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
    """Stage 3 RunnerHooks: every agent is real."""

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
        self._mass_transfer = build_mass_transfer_specialist(self._client, self._tools)
        self._metabolic = build_metabolic_specialist(self._client, self._tools)
        self._synthesizer = build_synthesizer(self._client)
        self._critic = build_critic(self._client, self._tools)
        self._judge = build_judge(self._client)
        self._specialists: dict[SpecialistRole, SpecialistAgent] = {
            "kinetics": self._kinetics,
            "mass_transfer": self._mass_transfer,
            "metabolic": self._metabolic,
        }

    # ---- orchestrator ----

    def pick_topic(self, state: RunnerState) -> str | None:
        view = project_orchestrator(
            events=(),
            seed_topics=list(state.seed_topics),
            budget=state.budget,
            current_turn=state.current_turn + 1,
        )
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
        agent = self._specialists.get(role)
        if agent is None:
            raise RuntimeError(f"no specialist registered for role {role!r}")
        view = project_specialist(
            events=(),
            role=role,
            current_topic=state.current_topic,
            available_findings=self._bundle.findings_pool,
            available_narratives=self._bundle.narratives_pool,
            available_trajectories=self._bundle.trajectories_pool,
            available_priors=self._bundle.priors_pool,
            available_analyses=self._bundle.analyses_pool,
        )
        result = agent.contribute(view, facet_id=facet_id)
        return result.facet, result.input_tokens, result.output_tokens

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

    # ---- critic ----

    def critique(self, state: RunnerState) -> tuple[CritiqueFull, int, int]:
        assert state.current_hypothesis is not None
        view = project_critic(
            hypothesis=state.current_hypothesis,
            citation_lookups=self._build_citation_lookups(state.current_hypothesis),
            relevant_priors=self._bundle.priors_pool,
        )
        result = self._critic.critique(view)
        return result.critique, result.input_tokens, result.output_tokens

    # ---- judge ----

    def judge(self, state: RunnerState) -> tuple[bool, str, int, int]:
        assert state.current_hypothesis is not None
        assert state.current_critique is not None
        view = project_judge(
            hypothesis=state.current_hypothesis,
            critique=state.current_critique,
            citation_lookups=self._build_citation_lookups(state.current_hypothesis),
        )
        result = self._judge.rule(view)
        return (
            result.criticism_valid,
            result.rationale,
            result.input_tokens,
            result.output_tokens,
        )

    # ---- helpers ----

    def _build_citation_lookups(self, hyp) -> dict:
        """Pre-resolve cited IDs so critic/judge don't waste calls re-querying."""
        lookups: dict = {}
        char = self._bundle.characterization
        finding_by_id = {f.finding_id: f for f in char.findings}
        narr_by_id = {n.narrative_id: n for n in char.narrative_observations}
        for fid in hyp.cited_finding_ids:
            f = finding_by_id.get(fid)
            if f is not None:
                lookups[fid] = {
                    "type": "finding",
                    "summary": f.summary,
                    "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                    "variables_involved": list(f.variables_involved),
                    "confidence": f.confidence,
                }
        for nid in hyp.cited_narrative_ids:
            n = narr_by_id.get(nid)
            if n is not None:
                lookups[nid] = {
                    "type": "narrative",
                    "tag": n.tag.value if hasattr(n.tag, "value") else str(n.tag),
                    "text": n.text,
                    "run_id": n.run_id,
                    "time_h": n.time_h,
                    "affected_variables": list(n.affected_variables),
                }
        return lookups
