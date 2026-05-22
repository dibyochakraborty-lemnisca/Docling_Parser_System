"""Projector — metric_id family routing.

When a Finding carries metric_id from the characterize-stage catalog,
project_specialist routes it to the matching specialist regardless of
whether the topic's affected_variables overlap the finding's variables.

The KINETICS / MASS_TRANSFER / METABOLIC families come from
fermdocs_characterize.agents.metric_catalog.
"""

from __future__ import annotations

from fermdocs_hypothesis.projector import (
    project_specialist,
    specialist_metric_ids,
)
from fermdocs_hypothesis.schema import FindingRef
from fermdocs_hypothesis.stubs.canned_agents import topic_spec_from_seed
from tests.unit.hypothesis.fixtures import make_seed_topic


def _project_for(role, findings):
    seed = make_seed_topic(affected_variables=["topic_only_var"])
    topic = topic_spec_from_seed(seed)
    return project_specialist(
        events=[],
        role=role,
        current_topic=topic,
        available_findings=findings,
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
    )


def test_specialist_metric_ids_for_known_roles_nonempty() -> None:
    assert "A8" in specialist_metric_ids("kinetics")
    assert "A14" in specialist_metric_ids("mass_transfer")
    assert "B10" in specialist_metric_ids("metabolic")


def test_specialist_metric_ids_unknown_role_empty() -> None:
    assert specialist_metric_ids("not_a_role") == frozenset()


def test_kinetics_finding_with_a8_routed_to_kinetics() -> None:
    findings = [
        FindingRef(
            finding_id="F-1",
            summary="mu_max=0.42 1/h",
            variables_involved=["biomass_g_l"],
            metric_id="A8",
        )
    ]
    view = _project_for("kinetics", findings)
    assert any(f.finding_id == "F-1" for f in view.relevant_findings)


def test_metabolic_finding_with_b10_routed_to_metabolic() -> None:
    findings = [
        FindingRef(
            finding_id="F-1",
            summary="RQ=1.42, overflow flag set",
            variables_involved=["our", "cer"],
            metric_id="B10",
        )
    ]
    view = _project_for("metabolic", findings)
    assert any(f.finding_id == "F-1" for f in view.relevant_findings)


def test_mass_transfer_finding_with_c4_routed_to_mass_transfer() -> None:
    findings = [
        FindingRef(
            finding_id="F-1",
            summary="kLa estimate 0.05 1/s",
            variables_involved=["agitation_rpm"],
            metric_id="C4",
        )
    ]
    view = _project_for("mass_transfer", findings)
    assert any(f.finding_id == "F-1" for f in view.relevant_findings)


def test_kinetics_metric_not_routed_to_metabolic() -> None:
    """A8 is kinetics-only; metabolic specialist's view shouldn't pick it
    up via metric_id family. (Other tag/variable matches could still pull
    it in — we test pure metric_id routing here by using a variable
    outside everyone's domain tags.)"""
    findings = [
        FindingRef(
            finding_id="F-1",
            summary="mu_max",
            variables_involved=["finding_only_var"],
            metric_id="A8",  # kinetics-only
        )
    ]
    view = _project_for("metabolic", findings)
    assert not any(f.finding_id == "F-1" for f in view.relevant_findings)


def test_finding_without_metric_id_falls_through_to_old_logic() -> None:
    """Backward compat: legacy findings (no metric_id) still go through
    the variable / tag / cited-id matchers and only the old logic
    decides whether they show up."""
    findings = [
        FindingRef(
            finding_id="F-1",
            summary="legacy finding",
            variables_involved=["finding_only_var"],
            metric_id=None,
        )
    ]
    view = _project_for("kinetics", findings)
    assert not any(f.finding_id == "F-1" for f in view.relevant_findings)
