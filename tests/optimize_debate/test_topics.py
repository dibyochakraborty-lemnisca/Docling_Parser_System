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


def test_discovered_levers_drive_topics_not_default_labs_knobs():
    from fermdocs_optimize.lever_discovery import Lever

    discovered = [
        Lever("nitrogen_source", "categorical", "metadata",
              {"R0": "CSL", "R1": "YE", "R2": "DBY"}),
        Lever("feed_g_l", "numeric", "metadata", {"R0": 5.0, "R1": 10.0, "R2": 15.0}),
    ]
    topics = extract_opportunity_topics(_char(), discovered_levers=discovered)
    knob_topics = [t for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION]
    # one topic per DISCOVERED lever, named after the real factor — not maltose/biomass
    assert [t.source_id for t in knob_topics] == ["lever-nitrogen_source", "lever-feed_g_l"]
    assert any("nitrogen_source" in t.summary for t in knob_topics)
    assert not any("maltose" in t.summary.lower() for t in knob_topics)
    # categorical lever lists its observed levels; metadata levers stay uncited
    cat = next(t for t in knob_topics if t.source_id == "lever-nitrogen_source")
    assert "CSL" in cat.summary and "YE" in cat.summary
    assert cat.cited_finding_ids == []  # design factor, no single measured channel


def test_derived_levers_are_dropped_metadata_only():
    # A2: only metadata design factors are debated; derived '.initial' channels
    # (incl. byproduct initials) are NOT debate topics, regardless of variation.
    from fermdocs_optimize.lever_discovery import Lever

    meta = [Lever("nitrogen_source", "categorical", "metadata", {"R0": "A", "R1": "B"}),
            Lever("feed_g_l", "numeric", "metadata", {"R0": 1.0, "R1": 2.0})]
    derived = [Lever("acetate_g_l.initial", "numeric", "derived", {"R0": 0.01, "R1": 0.33}),
               Lever("od600_au.initial", "numeric", "derived", {"R0": 1.08, "R1": 12.08})]
    topics = extract_opportunity_topics(_char(), discovered_levers=meta + derived)
    ids = {t.source_id for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION}
    assert ids == {"lever-nitrogen_source", "lever-feed_g_l"}
    # the byproduct initial is gone, even though it varied a lot
    assert "lever-acetate_g_l.initial" not in ids
    assert "lever-od600_au.initial" not in ids


def test_effect_size_attached_and_ranks_lever_above_flat_one():
    # The lever with the bigger measured titer swing must out-rank the flat one.
    from fermdocs_hypothesis.ranker import rank_topics
    from fermdocs_optimize.lever_discovery import Lever

    levers = [Lever("nitrogen_source", "categorical", "metadata", {"R0": "A", "R1": "B"}),
              Lever("flat_factor", "numeric", "metadata", {"R0": 1.0, "R1": 2.0})]
    effects = {"nitrogen_source": {"delta": 40.0, "n": 8, "best_setting": "B", "norm_effect": 0.9},
               "flat_factor": {"delta": 1.0, "n": 8, "best_setting": 2.0, "norm_effect": 0.02}}
    topics = extract_opportunity_topics(_char(), discovered_levers=levers, lever_effects=effects)
    nitro = next(t for t in topics if t.source_id == "lever-nitrogen_source")
    flat = next(t for t in topics if t.source_id == "lever-flat_factor")
    assert nitro.effect_size == 0.9 and flat.effect_size == 0.02
    assert "associates with +40" in nitro.summary  # effect grounded in the prompt
    ranked = rank_topics(topics, [], k=5)
    order = [r.topic_id for r in ranked]
    assert order.index(nitro.topic_id) < order.index(flat.topic_id)


def test_weak_association_is_labelled_in_topic_summary():
    from fermdocs_optimize.lever_discovery import Lever

    levers = [Lever("nitrogen_source", "categorical", "metadata", {"R0": "A", "R1": "B"}),
              Lever("strong_factor", "numeric", "metadata", {"R0": 1.0, "R1": 2.0})]
    # nitrogen: tiny effect (norm 0.03) -> WEAK; strong_factor: norm 0.9 -> not weak
    effects = {"nitrogen_source": {"delta": 2.0, "n": 10, "best_setting": "B", "norm_effect": 0.03},
               "strong_factor": {"delta": 40.0, "n": 10, "best_setting": 2.0, "norm_effect": 0.9}}
    topics = extract_opportunity_topics(_char(), discovered_levers=levers, lever_effects=effects)
    nitro = next(t for t in topics if t.source_id == "lever-nitrogen_source")
    strong = next(t for t in topics if t.source_id == "lever-strong_factor")
    assert "WEAK" in nitro.summary          # tiny effect flagged as preliminary
    assert "WEAK" not in strong.summary     # real effect not flagged


def test_no_discovered_levers_falls_back_to_static():
    # discovery did not run (None) -> static LABS lever set still used.
    topics = extract_opportunity_topics(_char(), discovered_levers=None)
    knob = [t for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION]
    assert len(knob) == len(DEFAULT_LEVERS)


def test_empty_metadata_runs_on_trends_not_labs_fallback():
    # data path ran but found only derived levers -> NO lever topics, trends carry.
    from fermdocs_optimize.lever_discovery import Lever

    derived = [Lever("acetate_g_l.initial", "numeric", "derived", {"R0": 0.01, "R1": 0.33})]
    topics = extract_opportunity_topics(_char(), discovered_levers=derived)
    knob = [t for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION]
    assert knob == []  # no lever topics, and crucially NOT the LABS DEFAULT_LEVERS
    assert any(t.source_type == TopicSourceType.TREND for t in topics)


def test_trend_topics_inherit_finding_severity_and_outrank_levers():
    from fermdocs_characterize.schema import Severity
    from fermdocs_optimize_debate.topics import _KNOB_PRIORITY, _TREND_PRIORITY

    # trends are evidence-grounded -> weighted above the speculative levers
    assert _TREND_PRIORITY > _KNOB_PRIORITY

    char = SimpleNamespace(
        findings=[
            _Finding("characterization:F-0009", "Titer plateaued early", ["P"]),
        ],
        trajectories=[],
    )
    # give the finding a severity the topic should inherit (was hardcoded MINOR)
    char.findings[0].severity = Severity.CRITICAL
    topics = extract_opportunity_topics(char)
    trend = next(t for t in topics if t.source_type == TopicSourceType.TREND)
    assert trend.severity == Severity.CRITICAL
    assert trend.priority == _TREND_PRIORITY


def test_higher_severity_trends_kept_when_capped():
    from fermdocs_characterize.schema import Severity

    findings = []
    for i, sev in enumerate([Severity.INFO, Severity.MINOR, Severity.CRITICAL, Severity.MAJOR]):
        f = _Finding(f"characterization:F-{i:04d}", f"obs {i}", ["P"])
        f.severity = sev
        findings.append(f)
    char = SimpleNamespace(findings=findings, trajectories=[])
    topics = extract_opportunity_topics(char, max_trend_topics=2)
    trend = [t for t in topics if t.source_type == TopicSourceType.TREND]
    assert len(trend) == 2
    # the two worst observations survive the cap
    assert {t.severity for t in trend} == {Severity.CRITICAL, Severity.MAJOR}


def test_ranker_effect_term_is_gated_and_weighted():
    # Guarantee: a topic with no effect_size (every diagnosis-stage topic) scores
    # exactly as before; the term only adds 0.6*effect_size when present.
    from fermdocs_characterize.schema import Severity
    from fermdocs_hypothesis.ranker import _EFFECT_WEIGHT, _score_seed
    from fermdocs_hypothesis.schema import SeedTopic, TopicSourceType

    def _topic(eid, eff):
        return SeedTopic(topic_id=eid, summary="x", source_type=TopicSourceType.OPEN_QUESTION,
                         source_id="s", severity=Severity.MAJOR, priority=0.7, effect_size=eff)
    import pytest
    base = _score_seed(_topic("T-0001", 0.0), unresolved=[], attempts=0, rejections=0)
    witheff = _score_seed(_topic("T-0002", 0.5), unresolved=[], attempts=0, rejections=0)
    assert witheff - base == pytest.approx(_EFFECT_WEIGHT * 0.5)   # exact gated contribution
    # diagnosis-stage default (no effect_size) is identical to omitting the term
    assert _score_seed(_topic("T-0003", 0.0), unresolved=[], attempts=0, rejections=0) == base


def test_regression_3cfc2aa6_real_design_lever_leads_not_byproduct_initial():
    # Reproduce the praaj run: run_conditions with a varying nitrogen design
    # factor + acetate/ethanol channels + per-run titer. Assert the byproduct
    # initials are NOT topics and the nitrogen design factor leads.
    import pandas as pd

    from fermdocs.analysis.cross_run import lever_effects
    from fermdocs_hypothesis.ranker import rank_topics
    from fermdocs_optimize.lever_discovery import discover_levers

    sources = ["CSL", "YE"]
    base = {"CSL": 95.0, "YE": 140.0}          # nitrogen source genuinely moves titer
    run_conditions, rows = {}, []
    for i in range(8):
        src = sources[i % 2]
        rid = f"B{i}"
        run_conditions[rid] = {"main_fermentation_nitrogen_source": {"value": src}}
        rows += [
            (rid, "product_g_l", 48.0, base[src] + (i % 2)),
            (rid, "acetate_g_l", 0.0, 0.01 + 0.03 * i),    # byproduct, varies
            (rid, "acetate_g_l", 48.0, 0.5 + 0.1 * i),
            (rid, "substrate_g_l", 0.0, 100.0 + i),        # input, varies
        ]
    dossier = {"run_conditions": run_conditions}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])

    levers = discover_levers(dossier, obs)
    effects = lever_effects(dossier, obs)
    topics = extract_opportunity_topics(_char(), discovered_levers=levers, lever_effects=effects)
    knob_ids = {t.source_id for t in topics if t.source_type == TopicSourceType.OPEN_QUESTION}

    # (1) byproduct initials are NOT debate topics
    assert "lever-acetate_g_l.initial" not in knob_ids
    assert "lever-substrate_g_l.initial" not in knob_ids
    # (2) the nitrogen design factor IS a topic
    assert "lever-main_fermentation_nitrogen_source" in knob_ids
    # (3) and it leads the ranking
    ranked = rank_topics(topics, [], k=5)
    assert ranked[0].topic_id == next(
        t.topic_id for t in topics
        if t.source_id == "lever-main_fermentation_nitrogen_source")


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
