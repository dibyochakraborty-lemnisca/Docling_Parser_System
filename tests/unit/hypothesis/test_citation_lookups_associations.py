"""Regression: the critic/judge must be able to RESOLVE within-run association
citations (WRA-*), or it flags every association-grounded hypothesis as a
hallucination and rejects the whole debate (run 61f0b3e1)."""
from __future__ import annotations

from types import SimpleNamespace

from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.schema import WithinRunAssociationRef


def _assoc(lever="main_fermentation_nitrogen_source"):
    return WithinRunAssociationRef(
        assoc_id=f"WRA-{lever}", lever=lever, summary=f"{lever} assoc", delta=2.026,
        direction="set_to", n=10, norm_effect=0.7, objective="product_g_l",
        best_setting="Yeast Extract - Leiber H")


def test_citation_lookups_resolves_within_run_associations():
    stub_self = SimpleNamespace(_bundle=SimpleNamespace(
        characterization=SimpleNamespace(findings=[], narrative_observations=[]),
        within_run_pool=[_assoc()],
    ))
    hyp = SimpleNamespace(
        cited_finding_ids=[], cited_narrative_ids=[],
        cited_association_ids=["WRA-main_fermentation_nitrogen_source"],
    )
    lookups = LiveHooks._build_citation_lookups(stub_self, hyp)
    entry = lookups.get("WRA-main_fermentation_nitrogen_source")
    assert entry is not None                       # resolves -> NOT a hallucination
    assert entry["type"] == "within_run_association"
    assert entry["delta"] == 2.026
    assert entry["best_setting"] == "Yeast Extract - Leiber H"


def test_unknown_association_id_stays_unresolved():
    stub_self = SimpleNamespace(_bundle=SimpleNamespace(
        characterization=SimpleNamespace(findings=[], narrative_observations=[]),
        within_run_pool=[_assoc()],
    ))
    hyp = SimpleNamespace(cited_finding_ids=[], cited_narrative_ids=[],
                          cited_association_ids=["WRA-made-up"])
    lookups = LiveHooks._build_citation_lookups(stub_self, hyp)
    assert "WRA-made-up" not in lookups            # genuinely unknown -> critic can flag it


def test_no_within_run_pool_is_graceful():
    # diagnosis stage: bundle has no within_run_pool -> no crash, no assoc entries.
    stub_self = SimpleNamespace(_bundle=SimpleNamespace(
        characterization=SimpleNamespace(findings=[], narrative_observations=[])))
    hyp = SimpleNamespace(cited_finding_ids=[], cited_narrative_ids=[],
                          cited_association_ids=["WRA-x"])
    assert LiveHooks._build_citation_lookups(stub_self, hyp) == {}
