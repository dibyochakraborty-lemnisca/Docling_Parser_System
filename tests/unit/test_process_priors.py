"""Plan A Stage 1: process priors loader and resolution tests.

No agent-behavior coverage in this stage — just the data model + lookup.
Stage 2 wires the tool; Stage 3 lands the prompt + validator changes.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from fermdocs.domain.process_priors import (
    PriorBound,
    ProcessPriors,
    ResolvedPrior,
    cached_priors,
    load_priors,
    priors_version,
    resolve_priors,
)


# ---------------------------------------------------------------------------
# Default priors file — sanity checks
# ---------------------------------------------------------------------------


def test_default_priors_file_loads() -> None:
    """The shipped process_priors.yaml must parse cleanly through Pydantic."""
    priors = load_priors()
    assert priors.version == "1.0"
    assert len(priors.organisms) >= 3, "expect penicillium + yeast + E. coli at minimum"


def test_default_priors_covers_tier_a_organisms() -> None:
    priors = load_priors()
    names = {o.name for o in priors.organisms}
    assert "Saccharomyces cerevisiae" in names
    assert "Escherichia coli" in names
    assert "Penicillium chrysogenum" in names


def test_default_priors_every_bound_has_source() -> None:
    """No prior ships without a citation. Required-source rule from §2."""
    priors = load_priors()
    for org in priors.organisms:
        for fam in org.process_families:
            for var_name, bound in fam.priors.items():
                assert bound.source.strip(), (
                    f"prior {org.name}/{fam.name}/{var_name} has empty source"
                )


def test_priors_version_helper_reads_top_level_version() -> None:
    assert priors_version() == "1.0"


def test_cached_priors_reuses_instance() -> None:
    a = cached_priors()
    b = cached_priors()
    assert a is b


# ---------------------------------------------------------------------------
# Custom-path loading + version errors
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(dedent(body).lstrip())


def test_load_priors_custom_path(tmp_path: Path) -> None:
    p = tmp_path / "custom.yaml"
    _write_yaml(
        p,
        """
        version: "1.0"
        organisms:
          - name: "Test organism"
            aliases: ["TO"]
            process_families:
              - name: "test_family"
                description: "stub"
                priors:
                  biomass_g_l:
                    range: [10.0, 20.0]
                    typical: 15.0
                    source: "Test 2026"
        """,
    )
    priors = load_priors(p)
    assert len(priors.organisms) == 1
    assert priors.organisms[0].name == "Test organism"


def test_priors_version_errors_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("organisms: []\n")
    with pytest.raises(ValueError, match="missing string `version`"):
        priors_version(p)


# ---------------------------------------------------------------------------
# PriorBound validation
# ---------------------------------------------------------------------------


def test_prior_bound_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="range low="):
        PriorBound(range=(5.0, 1.0), typical=3.0, source="Test")


def test_prior_bound_requires_source() -> None:
    with pytest.raises(ValueError):
        PriorBound(range=(0.0, 10.0), typical=5.0, source="")


# ---------------------------------------------------------------------------
# resolve_priors — filter combinations
# ---------------------------------------------------------------------------


@pytest.fixture
def priors() -> ProcessPriors:
    return load_priors()


def test_resolve_no_filter_returns_all_rows(priors: ProcessPriors) -> None:
    rows = resolve_priors(priors)
    # 3 organisms × at least a few priors each = clearly non-empty
    assert len(rows) > 5
    assert all(isinstance(r, ResolvedPrior) for r in rows)


def test_resolve_organism_substring_match(priors: ProcessPriors) -> None:
    """Case-insensitive substring on organism name + aliases."""
    yeast = resolve_priors(priors, organism="Saccharomyces cerevisiae")
    assert len(yeast) > 0
    assert all(r.organism == "Saccharomyces cerevisiae" for r in yeast)


def test_resolve_organism_alias_match(priors: ProcessPriors) -> None:
    """The dossier's messy 'E. coli BL21(DE3)' should still match."""
    rows = resolve_priors(priors, organism="E. coli BL21(DE3)")
    assert len(rows) > 0
    assert rows[0].organism == "Escherichia coli"


def test_resolve_organism_substring_within_alias(priors: ProcessPriors) -> None:
    """Query 'BL21' is a substring of an alias."""
    rows = resolve_priors(priors, organism="BL21")
    assert any(r.organism == "Escherichia coli" for r in rows)


def test_resolve_organism_no_match_returns_empty(priors: ProcessPriors) -> None:
    assert resolve_priors(priors, organism="Bacillus subtilis") == []


def test_resolve_process_family_filter(priors: ProcessPriors) -> None:
    rows = resolve_priors(
        priors,
        organism="Penicillium",
        process_family="submerged_fed_batch_paa",
    )
    assert len(rows) > 0
    assert all(r.process_family == "submerged_fed_batch_paa" for r in rows)


def test_resolve_process_family_case_insensitive(priors: ProcessPriors) -> None:
    rows = resolve_priors(
        priors,
        organism="Penicillium",
        process_family="SUBMERGED_FED_BATCH_PAA",
    )
    assert len(rows) > 0


def test_resolve_variable_filter(priors: ProcessPriors) -> None:
    rows = resolve_priors(priors, variable="biomass_endpoint_g_l")
    assert len(rows) >= 3, "every Tier A organism declares biomass_endpoint_g_l"
    assert all(r.variable == "biomass_endpoint_g_l" for r in rows)


def test_resolve_combined_filters(priors: ProcessPriors) -> None:
    rows = resolve_priors(
        priors,
        organism="S. cerevisiae",
        variable="ethanol_g_l",
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.organism == "Saccharomyces cerevisiae"
    assert r.variable == "ethanol_g_l"
    assert r.range == (0.0, 2.0)
    assert "Crabtree" in (r.note or "")


def test_resolve_returned_dict_is_json_safe(priors: ProcessPriors) -> None:
    rows = resolve_priors(priors, organism="E. coli", variable="acetate_g_l")
    assert len(rows) == 1
    payload = rows[0].to_dict()
    assert payload["range"] == [0.0, 2.0]
    assert payload["organism"] == "Escherichia coli"
    assert "Eiteman" in payload["source"]


# ---------------------------------------------------------------------------
# Domain content checks (sanity on actual prior values)
# ---------------------------------------------------------------------------


def test_yeast_mu_x_max_is_lower_than_ecoli(priors: ProcessPriors) -> None:
    """Crabtree-controlled yeast μ_max < E. coli μ_max — biologically required."""
    yeast = resolve_priors(priors, organism="S. cerevisiae", variable="mu_x_max_per_h")[0]
    ecoli = resolve_priors(priors, organism="E. coli", variable="mu_x_max_per_h")[0]
    assert yeast.range[1] < ecoli.range[0], (
        f"yeast max {yeast.range[1]} should be below E. coli min {ecoli.range[0]}"
    )


def test_penicillium_typical_mu_is_lowest(priors: ProcessPriors) -> None:
    """Filamentous Penicillium grows slower at the typical operating point.

    Range edges can overlap (high-end Penicillium ≈ low-end yeast); the
    biologically meaningful claim is that the typical/operating-point μ
    is ordered Penicillium < yeast < E. coli.
    """
    pen = resolve_priors(priors, organism="Penicillium", variable="mu_x_max_per_h")[0]
    yeast = resolve_priors(priors, organism="S. cerevisiae", variable="mu_x_max_per_h")[0]
    ecoli = resolve_priors(priors, organism="E. coli", variable="mu_x_max_per_h")[0]
    assert pen.typical < yeast.typical < ecoli.typical
