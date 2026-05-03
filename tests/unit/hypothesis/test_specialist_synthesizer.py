"""Specialist + Synthesizer agent — offline shape tests."""

from __future__ import annotations

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.agents.specialist_base import SpecialistAgent
from fermdocs_hypothesis.agents.synthesizer import SynthesizerAgent
from fermdocs_hypothesis.schema import (
    CitationCatalog,
    SpecialistView,
    SynthesizerView,
    TopicSpec,
)
from tests.unit.hypothesis.fixtures import CHAR_ID, make_facet, make_seed_topic
from fermdocs_hypothesis.stubs.canned_agents import topic_spec_from_seed


def test_specialist_build_facet_backfills_citations_when_llm_drops_them():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    view = SpecialistView(specialist_role="kinetics", current_topic=topic)
    agent = SpecialistAgent(client=None, spec={}, tools=None, role="kinetics")  # type: ignore[arg-type]
    parsed = {
        "action": "contribute_facet",
        "summary": "x",
        # no citations at all
    }
    facet = agent._build_facet(parsed, view, "FCT-0001")
    # backfilled from topic citations
    assert facet.cited_finding_ids == list(topic.cited_finding_ids)


def test_specialist_build_facet_clamps_confidence_at_cap():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    view = SpecialistView(specialist_role="kinetics", current_topic=topic)
    agent = SpecialistAgent(client=None, spec={}, tools=None, role="kinetics")  # type: ignore[arg-type]
    parsed = {
        "action": "contribute_facet",
        "summary": "x",
        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
        "confidence": 0.99,
    }
    facet = agent._build_facet(parsed, view, "FCT-0001")
    assert facet.confidence == 0.85


def test_specialist_build_facet_invalid_basis_falls_back_to_schema_only():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    view = SpecialistView(specialist_role="kinetics", current_topic=topic)
    agent = SpecialistAgent(client=None, spec={}, tools=None, role="kinetics")  # type: ignore[arg-type]
    parsed = {
        "action": "contribute_facet",
        "summary": "x",
        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
        "confidence": 0.6,
        "confidence_basis": "wrong_value",
    }
    facet = agent._build_facet(parsed, view, "FCT-0001")
    assert facet.confidence_basis == ConfidenceBasis.SCHEMA_ONLY


def test_synthesizer_build_hypothesis_includes_all_facet_ids():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    f1 = make_facet(facet_id="FCT-0001")
    f2 = make_facet(facet_id="FCT-0002")
    view = SynthesizerView(
        current_topic=topic,
        facets=[f1, f2],
        citation_universe=CitationCatalog(finding_ids=[f"{CHAR_ID}:F-0001"]),
    )
    agent = SynthesizerAgent(client=None)  # type: ignore[arg-type]
    parsed = {
        "summary": "synth",
        "facet_ids": ["FCT-0001"],  # LLM dropped FCT-0002
        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
        "confidence": 0.7,
        "confidence_basis": "schema_only",
    }
    hyp = agent._build_hypothesis(parsed, view, "H-0001")
    assert set(hyp.facet_ids) == {"FCT-0001", "FCT-0002"}


def test_synthesizer_caps_confidence_at_max_facet_confidence():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    f1 = make_facet(facet_id="FCT-0001")  # confidence 0.7 from fixture
    view = SynthesizerView(
        current_topic=topic,
        facets=[f1],
        citation_universe=CitationCatalog(finding_ids=[f"{CHAR_ID}:F-0001"]),
    )
    agent = SynthesizerAgent(client=None)  # type: ignore[arg-type]
    parsed = {
        "summary": "synth",
        "facet_ids": ["FCT-0001"],
        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
        "confidence": 0.85,  # tries to go above max facet
        "confidence_basis": "schema_only",
    }
    hyp = agent._build_hypothesis(parsed, view, "H-0001")
    assert hyp.confidence <= 0.7


def test_synthesizer_backfills_citations_from_universe():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    f1 = make_facet(facet_id="FCT-0001")
    view = SynthesizerView(
        current_topic=topic,
        facets=[f1],
        citation_universe=CitationCatalog(finding_ids=["F-A", "F-B"]),
    )
    agent = SynthesizerAgent(client=None)  # type: ignore[arg-type]
    parsed = {
        "summary": "synth",
        "facet_ids": ["FCT-0001"],
        # all citations dropped
        "confidence": 0.6,
        "confidence_basis": "schema_only",
    }
    hyp = agent._build_hypothesis(parsed, view, "H-0001")
    assert hyp.cited_finding_ids == ["F-A", "F-B"]
