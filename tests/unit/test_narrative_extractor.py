"""Plan B Stage 2.1: NarrativeExtractor unit tests.

No real Gemini calls — uses a scripted client that returns canned responses.
The real-PDF integration test lives in tests/integration/.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from fermdocs.domain.models import NarrativeBlock, NarrativeBlockType
from fermdocs.narrative import (
    NarrativeExtractor,
    extract_narrative_observations,
)
from fermdocs_characterize.schema import NarrativeObservation, NarrativeTag


CHAR_ID = uuid.UUID(int=314)


def _block(text: str, page: int = 1, idx: int = 0, section: str = "Results") -> NarrativeBlock:
    return NarrativeBlock(
        text=text,
        type=NarrativeBlockType.PARAGRAPH,
        locator={"page": page, "paragraph_idx": idx, "section": section, "format": "pdf"},
    )


class _ScriptedClient:
    def __init__(self, response: list[dict[str, Any]] | Exception) -> None:
        self._response = response
        self.calls: list[str] = []

    def call(self, rendered_blocks: str) -> list[dict[str, Any]]:
        self.calls.append(rendered_blocks)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


# ---------------------------------------------------------------------------
# Empty-input fast paths
# ---------------------------------------------------------------------------


def test_empty_blocks_returns_empty_no_call() -> None:
    client = _ScriptedClient([])
    out = NarrativeExtractor(client=client).extract([], characterization_id=CHAR_ID)
    assert out == []
    assert client.calls == []


def test_blocks_with_only_whitespace_returns_empty_no_call() -> None:
    client = _ScriptedClient([])
    blocks = [_block("   "), _block("\n\t")]
    out = NarrativeExtractor(client=client).extract(blocks, characterization_id=CHAR_ID)
    assert out == []
    # The renderer skipped both → empty rendered string → no LLM call
    assert client.calls == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_basic_extraction_namespaces_ids_and_preserves_fields() -> None:
    client = _ScriptedClient(
        [
            {
                "tag": "closure_event",
                "text": "terminated at 82h, white cells observed",
                "source_locator": {"page": 4, "section": "Results", "paragraph_index": 2},
                "run_id": "BATCH-01",
                "time_h": 82.0,
                "affected_variables": ["viability_pct"],
                "confidence": 0.8,
            },
            {
                "tag": "intervention",
                "text": "200 mL IPM added at 24h",
                "source_locator": {"page": 13, "section": "Procedure"},
                "run_id": "BATCH-05",
                "time_h": 24.0,
                "affected_variables": ["wcw_g_l"],
                "confidence": 0.85,
            },
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("dummy")], characterization_id=CHAR_ID
    )
    assert len(out) == 2
    assert out[0].narrative_id == f"{CHAR_ID}:N-0001"
    assert out[1].narrative_id == f"{CHAR_ID}:N-0002"
    assert out[0].tag == NarrativeTag.CLOSURE_EVENT
    assert out[1].tag == NarrativeTag.INTERVENTION
    assert out[0].run_id == "BATCH-01"
    assert out[0].time_h == 82.0
    assert out[1].source_locator.page == 13


def test_block_render_includes_index_page_section() -> None:
    """The rendered text the LLM sees must include the BLOCK index, page,
    and section so source_locator can be filled correctly."""
    client = _ScriptedClient([])
    blocks = [
        _block("first body", page=2, idx=0, section="Methods"),
        _block("second body", page=4, idx=1, section="Results"),
    ]
    NarrativeExtractor(client=client).extract(blocks, characterization_id=CHAR_ID)
    rendered = client.calls[0]
    assert "[BLOCK 0 | page=2 | section=Methods" in rendered
    assert "[BLOCK 1 | page=4 | section=Results" in rendered
    assert "first body" in rendered
    assert "second body" in rendered


# ---------------------------------------------------------------------------
# Defensive coercion
# ---------------------------------------------------------------------------


def test_unknown_tag_dropped() -> None:
    client = _ScriptedClient(
        [
            {"tag": "made_up_tag", "text": "ignored"},
            {"tag": "observation", "text": "kept"},
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert len(out) == 1
    assert out[0].text == "kept"
    assert out[0].narrative_id.endswith(":N-0001")


def test_empty_text_dropped() -> None:
    client = _ScriptedClient(
        [
            {"tag": "observation", "text": "   "},
            {"tag": "observation", "text": "real"},
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert [o.text for o in out] == ["real"]


def test_verbatim_dedup() -> None:
    client = _ScriptedClient(
        [
            {"tag": "observation", "text": "white cells observed"},
            {"tag": "observation", "text": "WHITE CELLS OBSERVED"},  # case-insensitive same
            {"tag": "observation", "text": "white cells observed"},  # same again
            {"tag": "observation", "text": "different statement"},
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    texts = [o.text for o in out]
    assert texts == ["white cells observed", "different statement"]


def test_confidence_clamped_at_0_85() -> None:
    client = _ScriptedClient([{"tag": "observation", "text": "x", "confidence": 0.99}])
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert out[0].confidence == 0.85


def test_confidence_default_when_missing() -> None:
    client = _ScriptedClient([{"tag": "observation", "text": "x"}])
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert 0.0 < out[0].confidence <= 0.85


def test_invalid_int_fields_become_none() -> None:
    client = _ScriptedClient(
        [
            {
                "tag": "observation",
                "text": "x",
                "source_locator": {"page": "not-a-page", "paragraph_index": "x"},
            }
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert out[0].source_locator.page is None
    assert out[0].source_locator.paragraph_index is None


# ---------------------------------------------------------------------------
# Error swallowing — extraction is additive
# ---------------------------------------------------------------------------


def test_client_exception_returns_empty_list() -> None:
    client = _ScriptedClient(RuntimeError("upstream failure"))
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert out == []


def test_client_returns_non_list_returns_empty() -> None:
    class _BadClient:
        def call(self, rendered_blocks: str):
            return {"not": "a list"}  # type: ignore[return-value]

    out = NarrativeExtractor(client=_BadClient()).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert out == []


def test_non_dict_items_skipped() -> None:
    client = _ScriptedClient(
        [
            "not a dict",  # type: ignore[list-item]
            None,  # type: ignore[list-item]
            {"tag": "observation", "text": "real"},
        ]
    )
    out = NarrativeExtractor(client=client).extract(
        [_block("x")], characterization_id=CHAR_ID
    )
    assert len(out) == 1
    assert out[0].text == "real"


# ---------------------------------------------------------------------------
# Document-size cap
# ---------------------------------------------------------------------------


def test_truncation_at_char_cap() -> None:
    """A document above the cap should still produce extractions from the
    portion that fits (truncation logged, not blocking)."""
    client = _ScriptedClient([{"tag": "observation", "text": "extracted"}])
    big_block = _block("x" * 5000)
    blocks = [big_block] * 200  # ~1MB rendered
    out = NarrativeExtractor(client=client, char_cap=20_000).extract(
        blocks, characterization_id=CHAR_ID
    )
    assert len(out) == 1
    assert len(client.calls[0]) <= 22_000  # respected cap (header overhead allowed)


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


def test_extract_narrative_observations_helper() -> None:
    client = _ScriptedClient([{"tag": "observation", "text": "hello"}])
    out = extract_narrative_observations(
        [_block("x")],
        characterization_id=CHAR_ID,
        client=client,
    )
    assert len(out) == 1
    assert isinstance(out[0], NarrativeObservation)


# ---------------------------------------------------------------------------
# Auto-namespacing for cross-instance ID stability
# ---------------------------------------------------------------------------


def test_separate_invocations_get_independent_seq_namespaces() -> None:
    """Each call starts seq at 1 — narrative_id is namespace-stable to the
    characterization_id, not to extractor instance."""
    client = _ScriptedClient([{"tag": "observation", "text": "a"}])
    extractor = NarrativeExtractor(client=client)
    out1 = extractor.extract([_block("x")], characterization_id=CHAR_ID)
    other = uuid.UUID(int=999)
    out2 = extractor.extract([_block("x")], characterization_id=other)
    assert out1[0].narrative_id == f"{CHAR_ID}:N-0001"
    assert out2[0].narrative_id == f"{other}:N-0001"
