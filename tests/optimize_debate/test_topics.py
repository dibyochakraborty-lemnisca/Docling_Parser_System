"""Phase A: opportunity-topic extraction + lever mapping (deterministic, no LLM).

Uses duck-typed stand-ins for characterization findings/trajectories so we never
construct a full CharacterizationOutput. Verifies the knob-anchored + trend
topics, their evidence citations, well-formed ids, and that debated levers map
back onto the optimizer's knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from fermdocs_hypothesis.schema import TopicSourceType

from fermdocs_optimize_debate.levers import DEFAULT_LEVERS, knobs_for_variables
from fermdocs_optimize_debate.schema import levers_from_output
from fermdocs_optimize_debate.topics import extract_opportunity_topics


@dataclass
class _Finding:
    finding_id: str
    summary: str
    variables_involved: list[str] = field(default_factory=list)


@dataclass
class _Traj:
    run_id: str
    variable: str


def _char():
    return SimpleNamespace(
        findings=[
            _Finding("characterization:F-0001", "Substrate left unconsumed at harvest", ["S", "P"]),
            _Finding("characterization:F-0002", "Dissolved oxygen dipped mid-run", ["DO"]),
            _Finding("characterization:F-0003", "Product titer plateaued early", ["P"]),
        ],
        trajectories=[
            _Traj("run-1", "S"), _Traj("run-1", "P"),
            _Traj("run-1", "X"), _Traj("run-1", "DO"),
        ],
    )


def test_one_knob_topic_per_lever():
    topics = extract_opportunity_topics(_char())
    knob_topics = [t for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION]
    assert len(knob_topics) == len(DEFAULT_LEVERS)
    assert [t.source_id for t in knob_topics] == [f"lever-{l.knob}" for l in DEFAULT_LEVERS]
    # ids are well-formed and unique
    ids = [t.topic_id for t in topics]
    assert all(i.startswith("T-") for i in ids)
    assert len(ids) == len(set(ids))


def test_knob_topic_cites_only_relevant_evidence():
    topics = extract_opportunity_topics(_char())
    biomass = next(t for t in topics if t.source_id == "lever-biomass")
    # biomass acts on X, P -> cites the P finding and the X/P trajectories, not the DO/S-only ones
    assert "characterization:F-0003" in biomass.cited_finding_ids   # P plateau
    assert "characterization:F-0002" not in biomass.cited_finding_ids  # DO only
    traj_vars = {tr.variable for tr in biomass.cited_trajectories}
    assert traj_vars <= {"X", "P"}


def test_trend_topics_prioritize_objective_and_are_capped():
    topics = extract_opportunity_topics(_char(), max_trend_topics=2)
    trend = [t for t in topics if t.source_type == TopicSourceType.TREND]
    assert len(trend) == 2
    # the DO-only finding doesn't touch any driver var -> excluded
    assert all("F-0002" not in t.source_id for t in trend)
    # an objective (P) finding sorts first
    assert "P" in trend[0].affected_variables


def test_no_topics_when_no_findings():
    empty = SimpleNamespace(findings=[], trajectories=[])
    topics = extract_opportunity_topics(empty)
    # still emits the knob levers (the box spine); just no evidence/trend topics
    assert len(topics) == len(DEFAULT_LEVERS)
    assert all(t.cited_finding_ids == [] for t in topics)


def test_knobs_for_variables_maps_back():
    assert knobs_for_variables(["X"]) == ["biomass"]
    assert "total_sub" in knobs_for_variables(["S"])
    assert knobs_for_variables(["DO"]) == []  # no knob acts on DO directly


def test_levers_from_output_sorts_and_maps():
    out = SimpleNamespace(final_hypotheses=[
        SimpleNamespace(hyp_id="H-0002", summary="raise substrate", affected_variables=["S", "P"],
                        actionable_recommendation="bump total_sub", confidence=0.6,
                        supporting_specialists=["kinetics"]),
        SimpleNamespace(hyp_id="H-0001", summary="unmappable", affected_variables=["pH"],
                        actionable_recommendation=None, confidence=0.9, supporting_specialists=[]),
    ])
    levers = levers_from_output(out)
    # knob-bearing lever sorts above the higher-confidence-but-unmappable one
    assert levers[0].lever_id == "H-0002"
    assert "total_sub" in levers[0].knobs
    assert levers[1].knobs == []  # pH maps to no knob, kept for narrative
