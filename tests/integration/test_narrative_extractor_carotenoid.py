"""Plan B Stage 2.1 integration test: real PDF → real Gemini extraction.

Auto-skips when GEMINI_API_KEY is missing or the carotenoid PDF is absent.
This is the eval gate for Stage 2 — extraction quality on industrial-style
prose. Hits the real API; runs as part of the suite when env permits.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from fermdocs_characterize.schema import NarrativeTag


CAROTENOID_PDF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "files"
    / "Carotenoid_batch_report_MASKED (1).pdf"
)

API_KEY_PRESENT = bool(os.environ.get("GEMINI_API_KEY"))


pytestmark = [
    pytest.mark.skipif(
        not API_KEY_PRESENT,
        reason="GEMINI_API_KEY not set; skipping live narrative extraction",
    ),
    pytest.mark.skipif(
        not CAROTENOID_PDF.exists(),
        reason=f"carotenoid PDF not present at {CAROTENOID_PDF}",
    ),
]


@pytest.fixture(scope="module")
def carotenoid_blocks():
    """Parse the carotenoid PDF once per module via the existing DoclingPdfParser."""
    from fermdocs.parsing.pdf_parser import DoclingPdfParser

    parser = DoclingPdfParser()
    result = parser.parse(CAROTENOID_PDF)
    assert result.narrative_blocks, "DoclingPdfParser produced no narrative blocks"
    return result.narrative_blocks


def test_extracts_at_least_one_closure_event(carotenoid_blocks) -> None:
    """The carotenoid report explicitly states cell death / white cells in
    every batch's closure. The extractor must surface at least one
    closure_event tagged observation."""
    from fermdocs.narrative import NarrativeExtractor

    char_id = uuid.UUID(int=1)
    obs = NarrativeExtractor().extract(
        carotenoid_blocks, characterization_id=char_id
    )
    assert obs, "extractor returned no observations from the carotenoid PDF"
    closures = [o for o in obs if o.tag == NarrativeTag.CLOSURE_EVENT]
    assert closures, (
        f"no closure_event tagged observations found. Tags emitted:"
        f" {[o.tag.value for o in obs]}"
    )
    # At least one closure event should mention cell death or white cells
    assert any(
        ("white cell" in o.text.lower() or "cell death" in o.text.lower())
        for o in closures
    ), f"closure events did not mention cell death / white cells: {[o.text for o in closures]}"


def test_extracts_at_least_one_intervention_for_ipm(carotenoid_blocks) -> None:
    """BATCH-05 and BATCH-06 add Isopropyl Myristate (IPM) to the reactor.
    The extractor must catch at least one intervention referencing IPM."""
    from fermdocs.narrative import NarrativeExtractor

    char_id = uuid.UUID(int=2)
    obs = NarrativeExtractor().extract(
        carotenoid_blocks, characterization_id=char_id
    )
    interventions = [o for o in obs if o.tag == NarrativeTag.INTERVENTION]
    # Loosened to "any tag" matching IPM since 'intervention' vs 'protocol_note'
    # is a tag-quality concern we iterate on; the *factual capture* is the gate.
    ipm_mentions = [o for o in obs if "ipm" in o.text.lower() or "isopropyl myristate" in o.text.lower()]
    assert ipm_mentions, (
        f"no observation mentions IPM / Isopropyl Myristate. Got tags:"
        f" {[(o.tag.value, o.text[:60]) for o in obs]}"
    )
    # Most ideal: at least one of the IPM mentions is tagged intervention
    assert any(o.tag == NarrativeTag.INTERVENTION for o in ipm_mentions), (
        f"IPM mentioned but not tagged 'intervention'; tags were:"
        f" {[(o.tag.value, o.text[:60]) for o in ipm_mentions]}"
    )


def test_observations_are_namespaced_to_characterization_id(carotenoid_blocks) -> None:
    """Sanity: every emitted narrative_id starts with the namespace prefix
    so CharacterizationOutput's namespace validator will accept them."""
    from fermdocs.narrative import NarrativeExtractor

    char_id = uuid.UUID(int=3)
    obs = NarrativeExtractor().extract(
        carotenoid_blocks, characterization_id=char_id
    )
    prefix = f"{char_id}:N-"
    for o in obs:
        assert o.narrative_id.startswith(prefix), (
            f"narrative_id {o.narrative_id!r} not namespaced to {char_id}"
        )


def test_observations_carry_extraction_model(carotenoid_blocks) -> None:
    from fermdocs.narrative import NarrativeExtractor

    obs = NarrativeExtractor().extract(
        carotenoid_blocks, characterization_id=uuid.UUID(int=4)
    )
    if not obs:
        pytest.skip("extractor returned empty list — separate test gates this")
    assert all(o.extraction_model for o in obs), (
        "every observation must carry the extraction_model that emitted it"
    )
