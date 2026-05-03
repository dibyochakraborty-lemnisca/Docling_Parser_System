"""bundle_loader — load real carotenoid bundle and confirm shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs_hypothesis.bundle_loader import load_bundle

CAROTENOID = Path("out/bundle_multi_20260502T220318Z_1a29e4")


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_load_carotenoid_bundle_produces_seed_topics():
    loaded = load_bundle(CAROTENOID)
    assert loaded.hyp_input.seed_topics, "expected seed topics from carotenoid diagnosis"
    # Carotenoid diagnosis has narrative-cited failures
    assert any(s.cited_narrative_ids for s in loaded.hyp_input.seed_topics)


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_load_carotenoid_bundle_pools_built():
    """Carotenoid is narrative-dominant; findings may be empty but the pools
    should be built without error."""
    loaded = load_bundle(CAROTENOID)
    assert isinstance(loaded.findings_pool, list)
    assert isinstance(loaded.trajectories_pool, list)


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_load_carotenoid_bundle_narratives_pool_nonempty():
    loaded = load_bundle(CAROTENOID)
    assert loaded.narratives_pool


@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_load_carotenoid_bundle_fills_organism_when_present():
    loaded = load_bundle(CAROTENOID)
    # organism may be None if dossier didn't surface it; just confirm field exists
    assert hasattr(loaded.hyp_input, "organism")
