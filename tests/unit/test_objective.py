"""resolve_objective: derive the optimization objective from the data + the user's
question, with no hardcoded channel constant (de-LABS, 2026-06-16)."""
from __future__ import annotations

from types import SimpleNamespace

from fermdocs.analysis.objective import resolve_objective
from fermdocs.domain.golden_schema import cached_schema


def test_golden_schema_designates_product_as_objective():
    # The golden schema is the single source of truth for the default objective.
    assert cached_schema().objective_channel() == "product_g_l"


def test_default_is_golden_objective_when_present():
    chans = {"product_g_l", "substrate_g_l", "od600_au"}
    assert resolve_objective(chans) == "product_g_l"


def test_user_question_overrides_default():
    uq = SimpleNamespace(affected_variables=["ethanol_g_l"])
    # the user is asking about ethanol -> ethanol is the objective, not product
    assert resolve_objective({"product_g_l", "ethanol_g_l"}, user_question=uq) == "ethanol_g_l"


def test_user_target_must_be_measured():
    # a user-named channel that isn't in the data is ignored -> fall back to default
    uq = SimpleNamespace(affected_variables=["not_measured_g_l"])
    assert resolve_objective({"product_g_l"}, user_question=uq) == "product_g_l"


def test_refuses_when_no_objective_resolvable():
    # no product channel + no user target -> None (caller must refuse, not guess)
    assert resolve_objective({"od600_au", "ph"}) is None


def test_refuses_on_empty_channels():
    assert resolve_objective(set()) is None
    assert resolve_objective([]) is None


def test_no_labs_species_default():
    # The old LABS default "P" must never be returned for a real channel set.
    assert resolve_objective({"product_g_l", "od600_au"}) != "P"
