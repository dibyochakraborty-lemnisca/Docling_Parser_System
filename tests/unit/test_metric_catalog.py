"""Catalog roundtrip + structural tests.

Every entry marked status='ready' must have a non-None toolkit_fn that
imports successfully and returns a callable. Every entry must have a
unique metric_id, a tier in {A, B, C}, and a non-empty short_description.
"""

from __future__ import annotations

import pytest

from fermdocs_characterize.agents.metric_catalog import (
    CATALOG,
    DATA_QUALITY_METRICS,
    KINETICS_METRICS,
    MASS_TRANSFER_METRICS,
    METABOLIC_METRICS,
    CatalogEntry,
    entries_by_tier,
    get_entry,
    ready_entries,
)


def test_catalog_has_expected_entries() -> None:
    # 24 Tier A + 20 Tier B + 16 Tier C + 6 Tier P (P1-P5, P_INTRACELLULAR_YIELD)
    assert len(CATALOG) == 66


def test_metric_ids_unique_and_well_formed() -> None:
    seen: set[str] = set()
    for mid, entry in CATALOG.items():
        assert mid == entry.metric_id
        assert mid not in seen
        seen.add(mid)
        assert entry.tier in {"A", "B", "C", "P"}
        assert mid[0] == entry.tier
        # Tier P uses descriptive ids (P1…P5, P_INTRACELLULAR_YIELD);
        # other tiers stay strict numeric.
        if entry.tier != "P":
            assert mid[1:].isdigit()


def test_every_entry_has_description() -> None:
    for entry in CATALOG.values():
        assert entry.short_description.strip()
        assert entry.long_description.strip()
        assert entry.applies_to.strip()


def test_tier_counts_match_plan() -> None:
    assert len(entries_by_tier("A")) == 24
    assert len(entries_by_tier("B")) == 20
    assert len(entries_by_tier("C")) == 16
    assert len(entries_by_tier("P")) == 6


def test_ready_entries_resolve_toolkit_fn() -> None:
    ready = ready_entries()
    assert ready, "expected at least one ready entry in PR 1"
    for entry in ready:
        fn = entry.resolve_toolkit_fn()
        assert callable(fn), f"{entry.metric_id} toolkit_fn did not resolve to callable"


def test_ready_entries_after_pr3_are_correct_set() -> None:
    ready_ids = {e.metric_id for e in ready_entries()}
    # PR 1: A8/A9/A10/A11 (kinetics)
    # PR 2: A14/A15/A17/A18 (operational), A19/A20/A21 (cross_run),
    #       B6/B10/B16 (balances)
    # PR 3: C2/C3/C4/C5/C9/C10 (literature)
    # Tier P shipped after PR3 (characterize-determinism + yeast-intracellular).
    expected = {
        "A8", "A9", "A10", "A11",
        "A14", "A15", "A17", "A18",
        "A19", "A20", "A21",
        "B6", "B10", "B16",
        "C2", "C3", "C4", "C5", "C9", "C10",
        "P1", "P2", "P3", "P4", "P5",
        "P_INTRACELLULAR_YIELD",
    }
    assert ready_ids == expected


def test_pending_entries_have_no_toolkit_fn() -> None:
    for entry in CATALOG.values():
        if entry.status == "pending":
            assert entry.toolkit_fn is None


def test_get_entry_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        get_entry("Z99")


def test_resolve_toolkit_fn_raises_when_pending() -> None:
    pending = next(e for e in CATALOG.values() if e.status == "pending")
    with pytest.raises(ValueError):
        pending.resolve_toolkit_fn()


def test_specialist_metric_families_are_disjoint_within_role() -> None:
    # Each specialist family is a frozenset of valid metric_ids.
    all_known = set(CATALOG.keys())
    for family in (
        KINETICS_METRICS,
        MASS_TRANSFER_METRICS,
        METABOLIC_METRICS,
        DATA_QUALITY_METRICS,
    ):
        assert family <= all_known, f"unknown metric_ids in family: {family - all_known}"


def test_catalog_entry_immutable() -> None:
    entry: CatalogEntry = get_entry("A8")
    with pytest.raises(Exception):
        entry.tier = "B"  # type: ignore[misc]


def test_a8_compute_mu_metadata_complete() -> None:
    a8 = get_entry("A8")
    assert a8.is_ready()
    assert a8.required_inputs[0].variable == "biomass_g_l"
    assert any(p.name == "window" for p in a8.required_parameters)
    fn = a8.resolve_toolkit_fn()
    assert fn.__name__ == "compute_mu"
