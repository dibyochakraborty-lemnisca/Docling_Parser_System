"""tools_bundle — query/get_priors/get_narrative_observations on real bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.tools_bundle.factory import (
    GET_NARRATIVE_OBSERVATIONS,
    GET_PRIORS,
    QUERY_BUNDLE,
    make_tool_bundle,
)

CAROTENOID = Path("out/bundle_multi_20260502T220318Z_1a29e4")


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_query_bundle_finding_substring():
    loaded = load_bundle(CAROTENOID)
    tb = make_tool_bundle(loaded)
    # Carotenoid bundle has narrative-heavy diagnosis; query with a generic
    # substring that should hit at least one narrative if not finding.
    res = tb.dispatch(QUERY_BUNDLE, {"scope": "narrative", "id_or_query": "BATCH"})
    assert "hits" in res or "narrative" in res


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_get_narrative_observations_returns_results():
    loaded = load_bundle(CAROTENOID)
    tb = make_tool_bundle(loaded)
    res = tb.dispatch(GET_NARRATIVE_OBSERVATIONS, {"limit": 5})
    assert "results" in res
    assert res["count"] <= 5


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_get_priors_with_yeast_returns_results():
    loaded = load_bundle(CAROTENOID)
    tb = make_tool_bundle(loaded)
    res = tb.get_priors(organism="Saccharomyces cerevisiae")
    assert "results" in res
    # If priors are loaded for yeast we should see some
    assert isinstance(res["results"], list)


def test_query_bundle_unknown_scope_returns_error_with_valid_list():
    # Use a minimally-valid loaded bundle wrapper: tests don't need a real one
    # for this branch since the dispatcher checks scope first. Skip if no bundle.
    if not CAROTENOID.exists():
        pytest.skip("carotenoid bundle not present")
    loaded = load_bundle(CAROTENOID)
    tb = make_tool_bundle(loaded)
    res = tb.query_bundle("invalid_scope", "anything")
    assert "error" in res
    assert "valid_scopes" in res


def test_dispatch_unknown_tool_returns_error():
    if not CAROTENOID.exists():
        pytest.skip("carotenoid bundle not present")
    loaded = load_bundle(CAROTENOID)
    tb = make_tool_bundle(loaded)
    res = tb.dispatch("not_a_tool", {})
    assert "error" in res
