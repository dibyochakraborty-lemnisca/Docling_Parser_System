"""Tests for build_segmenter and the Gemini segmenter client wiring.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Network is never touched — Gemini's `.call()` requires google.genai which
may not be importable in CI; we verify wiring (factory routes correctly,
client carries the right model name, schema is well-formed) without
invoking the LLM.
"""

from __future__ import annotations

import os

import pytest

from fermdocs.mapping.factory import UnknownProviderError, build_segmenter
from fermdocs.parsing.document_segmenter import DocumentSegmenter
from fermdocs.parsing.gemini_segmenter_client import (
    GeminiSegmenterClient,
    _DEFAULT_MODEL,
    _GEMINI_SEGMENTER_SCHEMA,
)


# ---------- factory ----------


def test_build_segmenter_fake_returns_none_client():
    """Fake/none provider yields a segmenter whose .segment() returns None
    immediately, so the pipeline falls back to existing chain.
    """
    seg = build_segmenter(provider="fake")
    assert isinstance(seg, DocumentSegmenter)
    assert seg._client is None  # type: ignore[attr-defined]


def test_build_segmenter_none_returns_none_client():
    seg = build_segmenter(provider="none")
    assert seg._client is None  # type: ignore[attr-defined]


def test_build_segmenter_gemini_returns_real_client():
    seg = build_segmenter(provider="gemini")
    assert isinstance(seg, DocumentSegmenter)
    assert isinstance(seg._client, GeminiSegmenterClient)  # type: ignore[attr-defined]


def test_build_segmenter_unknown_provider_raises():
    with pytest.raises(UnknownProviderError, match="unknown segmenter provider"):
        build_segmenter(provider="psychic")


def test_build_segmenter_resolves_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FERMDOCS_SEGMENTER_PROVIDER", "fake")
    monkeypatch.delenv("FERMDOCS_MAPPER_PROVIDER", raising=False)
    seg = build_segmenter(provider=None)
    assert seg._client is None  # type: ignore[attr-defined]


def test_build_segmenter_falls_back_to_mapper_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FERMDOCS_SEGMENTER_PROVIDER", raising=False)
    monkeypatch.setenv("FERMDOCS_MAPPER_PROVIDER", "fake")
    seg = build_segmenter(provider=None)
    assert seg._client is None  # type: ignore[attr-defined]


# ---------- GeminiSegmenterClient wiring ----------


def test_gemini_client_uses_default_model_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FERMDOCS_SEGMENTER_MODEL", raising=False)
    client = GeminiSegmenterClient()
    assert client.model_name == _DEFAULT_MODEL
    assert _DEFAULT_MODEL == "gemini-3.1-pro-preview"


def test_gemini_client_respects_env_model_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FERMDOCS_SEGMENTER_MODEL", "gemini-3-flash")
    client = GeminiSegmenterClient()
    assert client.model_name == "gemini-3-flash"


def test_gemini_client_respects_explicit_model_arg():
    client = GeminiSegmenterClient(model="some-other-model")
    assert client.model_name == "some-other-model"


def test_factory_propagates_gemini_model_name(monkeypatch: pytest.MonkeyPatch):
    """The DocumentSegmenter must carry the SAME model name the Gemini client
    is using, so DocumentMap.llm_model is accurate provenance."""
    monkeypatch.setenv("FERMDOCS_SEGMENTER_MODEL", "test-model-abc")
    seg = build_segmenter(provider="gemini")
    assert seg._model_name == "test-model-abc"  # type: ignore[attr-defined]
    assert seg._provider == "gemini"  # type: ignore[attr-defined]


# ---------- structured-output schema sanity ----------


def test_schema_has_required_top_level_fields():
    """The schema must enforce shape so the LLM can't omit critical fields."""
    schema = _GEMINI_SEGMENTER_SCHEMA
    assert schema["type"] == "OBJECT"
    assert set(schema["required"]) == {
        "runs",
        "unassigned_table_indices",
        "overall_confidence",
    }


def test_schema_run_object_required_fields():
    run_schema = _GEMINI_SEGMENTER_SCHEMA["properties"]["runs"]["items"]
    required = set(run_schema["required"])
    # rationale is optional (LLM may skip), all others must be present
    assert {
        "run_id",
        "display_name",
        "table_indices",
        "source_signal",
        "confidence",
    }.issubset(required)


def test_schema_source_signal_enum_matches_runsegmentsource():
    """Drift check: schema enum must match the RunSegmentSource enum
    values exactly. If a new RunSegmentSource value is added, the schema
    enum must be updated too."""
    from fermdocs.domain.models import RunSegmentSource

    run_schema = _GEMINI_SEGMENTER_SCHEMA["properties"]["runs"]["items"]
    schema_enum = set(run_schema["properties"]["source_signal"]["enum"])
    code_enum = {v.value for v in RunSegmentSource}
    assert schema_enum == code_enum, (
        "Schema source_signal enum drifted from RunSegmentSource code enum. "
        f"Schema: {schema_enum}, code: {code_enum}. "
        "Update _GEMINI_SEGMENTER_SCHEMA when adding/removing RunSegmentSource values."
    )


def test_schema_does_not_include_segmenter_owned_fields():
    """file_id, llm_model, llm_provider, schema_version are set by the
    segmenter, not by the LLM. Schema must not require them — we don't
    want the LLM hallucinating values for fields it doesn't control.
    """
    top_props = set(_GEMINI_SEGMENTER_SCHEMA["properties"].keys())
    forbidden = {"file_id", "llm_model", "llm_provider", "schema_version"}
    leaked = top_props & forbidden
    assert not leaked, f"schema leaks segmenter-owned fields: {leaked}"
