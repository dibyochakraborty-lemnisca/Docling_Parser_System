"""Phase B grounding tools: query_relationship + constrained execute_python.

These let an opportunity-debate specialist TEST a claim against the data instead
of asserting it (the "specialists defer because they can't compute" fix).
"""
from __future__ import annotations

from types import SimpleNamespace

from fermdocs_hypothesis.schema import WithinRunAssociationRef
from fermdocs_hypothesis.tools_bundle.factory import (
    EXECUTE_PYTHON,
    QUERY_RELATIONSHIP,
    HypothesisToolBundle,
)


def _assoc(lever="nitrogen_source"):
    return WithinRunAssociationRef(
        assoc_id=f"WRA-{lever}", lever=lever, summary=f"{lever} assoc", delta=40.0,
        direction="set_to", n=8, norm_effect=0.9, objective="product_g_l",
        best_setting="YE")


def _bundle(*, pool=None, bundle_dir=None):
    # HypothesisToolBundle is a dataclass over `bundle`; a namespace is enough
    # for the two tools (they getattr within_run_pool / bundle_dir).
    return HypothesisToolBundle(
        bundle=SimpleNamespace(within_run_pool=pool or [], bundle_dir=bundle_dir))


def test_query_relationship_returns_known_lever():
    tb = _bundle(pool=[_assoc("nitrogen_source"), _assoc("feed_g_l")])
    out = tb.dispatch(QUERY_RELATIONSHIP, {"lever": "nitrogen_source"})
    assert out["assoc_id"] == "WRA-nitrogen_source"
    assert out["delta"] == 40.0 and out["best_setting"] == "YE"
    assert out["n"] == 8


def test_query_relationship_unknown_lists_known():
    tb = _bundle(pool=[_assoc("nitrogen_source")])
    out = tb.dispatch(QUERY_RELATIONSHIP, {"lever": "made_up"})
    assert "error" in out
    assert out["known_levers"] == ["nitrogen_source"]


def test_query_relationship_empty_pool_is_graceful():
    out = _bundle().dispatch(QUERY_RELATIONSHIP, {"lever": "x"})
    assert "error" in out and out["known_levers"] == []


def test_execute_python_runs_in_sandbox():
    # no bundle_dir -> obs is None; the sandbox still runs arbitrary analysis.
    out = _bundle().dispatch(EXECUTE_PYTHON, {"code": "print(6 * 7)\nprint(obs is None)"})
    assert "42" in out["output"]
    assert "True" in out["output"]          # obs preloaded as None when no bundle
    assert out["timed_out"] is False


def test_execute_python_reports_errors_without_crashing():
    out = _bundle().dispatch(EXECUTE_PYTHON, {"code": "raise ValueError('boom')"})
    assert "boom" in out["output"] or out["returncode"] != 0
