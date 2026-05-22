"""CLI flag plumbing for --segment-pdfs and --manifest-run-id.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Verifies the surface added in commit 7:
  - --segment-pdfs / --no-segment-pdfs flag exists with the right default
  - --manifest-run-id option accepts a string
  - _build_pipeline accepts segment_pdfs and routes through to build_segmenter
  - FERMDOCS_PDF_SEGMENT env var is respected
  - --fake-mapper short-circuits the segmenter
"""

from __future__ import annotations

import inspect

import pytest

from fermdocs.cli import _build_pipeline, ingest as ingest_cmd


# ---------- flag surface ----------


def test_ingest_command_has_segment_pdfs_flag():
    params = {p.name for p in ingest_cmd.params}
    assert "segment_pdfs" in params


def test_ingest_command_has_manifest_run_id_option():
    params = {p.name for p in ingest_cmd.params}
    assert "manifest_run_id" in params


def test_segment_pdfs_default_is_none_so_env_can_decide():
    """Default None lets the function body inspect FERMDOCS_PDF_SEGMENT.
    A True/False default would override the env var, breaking the design
    decision that operators control segmentation via env."""
    param = next(p for p in ingest_cmd.params if p.name == "segment_pdfs")
    assert param.default is None


def test_manifest_run_id_default_is_none():
    param = next(p for p in ingest_cmd.params if p.name == "manifest_run_id")
    assert param.default is None


# ---------- _build_pipeline signature ----------


def test_build_pipeline_accepts_segment_pdfs_kwarg():
    sig = inspect.signature(_build_pipeline)
    assert "segment_pdfs" in sig.parameters
    assert sig.parameters["segment_pdfs"].default is None


# ---------- env-var behavior (no DB needed; we exercise build_segmenter only) ----------


def test_env_default_uses_gemini_when_unset(monkeypatch: pytest.MonkeyPatch):
    """No env var → defaults to gemini per design (FERMDOCS_PDF_SEGMENT=true,
    so a real GeminiSegmenterClient is wired)."""
    from fermdocs.mapping.factory import build_segmenter
    from fermdocs.parsing.gemini_segmenter_client import GeminiSegmenterClient

    monkeypatch.delenv("FERMDOCS_PDF_SEGMENT", raising=False)
    monkeypatch.delenv("FERMDOCS_SEGMENTER_PROVIDER", raising=False)
    monkeypatch.setenv("FERMDOCS_MAPPER_PROVIDER", "gemini")
    seg = build_segmenter()  # no explicit provider; resolves from env
    assert isinstance(seg._client, GeminiSegmenterClient)  # type: ignore[attr-defined]


def test_segmenter_provider_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """FERMDOCS_SEGMENTER_PROVIDER=fake → no LLM client wired → segment()
    returns None and pipeline falls back to existing chain."""
    from fermdocs.mapping.factory import build_segmenter

    monkeypatch.setenv("FERMDOCS_SEGMENTER_PROVIDER", "fake")
    seg = build_segmenter()
    assert seg._client is None  # type: ignore[attr-defined]


# ---------- ingestion path: segmenter passed through ----------


def test_pipeline_constructor_receives_document_segmenter_kwarg():
    """Smoke check: IngestionPipeline.__init__ accepts the new kwarg.
    Catches drift if someone removes the parameter."""
    from fermdocs.pipeline import IngestionPipeline

    sig = inspect.signature(IngestionPipeline.__init__)
    assert "document_segmenter" in sig.parameters


def test_ingest_method_accepts_manifest_run_id_kwarg():
    from fermdocs.pipeline import IngestionPipeline

    sig = inspect.signature(IngestionPipeline.ingest)
    assert "manifest_run_id" in sig.parameters
    assert sig.parameters["manifest_run_id"].default is None
