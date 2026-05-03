"""Schema invariants — id shapes, citation discipline, dup rejection."""

from __future__ import annotations

import pytest

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.schema import (
    CritiqueFull,
    FacetFull,
    HypothesisFull,
    OpenQuestionRef,
    SeedTopic,
)
from tests.unit.hypothesis.fixtures import (
    CHAR_ID,
    make_facet,
    make_hypothesis,
    make_seed_topic,
)


def test_seed_topic_id_shape():
    with pytest.raises(ValueError, match="topic_id must match"):
        make_seed_topic(topic_id="bad")


def test_facet_must_cite_something():
    with pytest.raises(ValueError, match="must cite"):
        FacetFull(
            facet_id="FCT-0001",
            specialist="kinetics",
            summary="x",
            cited_finding_ids=[],
            cited_narrative_ids=[],
            cited_trajectories=[],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        )


def test_hypothesis_must_cite_something():
    with pytest.raises(ValueError, match="must cite"):
        HypothesisFull(
            hyp_id="H-0001",
            summary="x",
            facet_ids=["FCT-0001"],
            cited_finding_ids=[],
            cited_narrative_ids=[],
            cited_trajectories=[],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        )


def test_red_flag_critique_requires_reasons():
    with pytest.raises(ValueError, match="red-flag"):
        CritiqueFull(hyp_id="H-0001", flag="red", reasons=[])


def test_green_flag_critique_no_reasons_required():
    crit = CritiqueFull(hyp_id="H-0001", flag="green", reasons=[])
    assert crit.flag == "green"


def test_open_question_ref_qid_shape():
    with pytest.raises(ValueError, match="qid must match"):
        OpenQuestionRef(qid="bad", question="x", raised_by="kinetics")


def test_open_question_ref_resolved_needs_resolution():
    with pytest.raises(ValueError, match="resolved=True requires"):
        OpenQuestionRef(qid="Q-0001", question="x", raised_by="kinetics", resolved=True)


def test_seed_topic_priority_bounds():
    with pytest.raises(ValueError):
        make_seed_topic(priority=1.5)
