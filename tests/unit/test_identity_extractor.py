"""Tests for IdentityExtractor: every branch in the two-layer decision flow.

The extractor returns a ProcessIdentity wrapping two independent layers:
  - observed:    surface facts (organism/product/scale/family hint)
  - registered:  registry classification

Failure of one layer must NOT nullify the other. These tests assert that
property explicitly: the yeast case below is the canonical "registered
fails, observed survives" example.
"""

from __future__ import annotations

from typing import Any

import pytest

from fermdocs.domain.models import (
    IdentityProvenance,
    NarrativeBlock,
    NarrativeBlockType,
)
from fermdocs.mapping.evidence_gated_llm import LLM_CONFIDENCE_CAP
from fermdocs.mapping.identity_extractor import IdentityExtractor
from fermdocs.mapping.process_registry import load_registry


@pytest.fixture
def registry():
    return load_registry()


@pytest.fixture
def penicillin_blocks():
    return [
        NarrativeBlock(
            text="Penicillium chrysogenum was cultured in fed-batch mode for penicillin production with PAA feed.",
            type=NarrativeBlockType.PARAGRAPH,
            locator={"paragraph_idx": 0, "page": 1, "file_id": "F1"},
        )
    ]


@pytest.fixture
def yeast_blocks():
    return [
        NarrativeBlock(
            text="Saccharomyces cerevisiae was used for ethanol production in batch mode.",
            type=NarrativeBlockType.PARAGRAPH,
            locator={"paragraph_idx": 0, "page": 1, "file_id": "F2"},
        )
    ]


@pytest.fixture
def fingerprint_present():
    """Variable set that satisfies penicillin_indpensim's fingerprint."""
    return {"paa_mg_l", "nh3_mg_l", "biomass_g_l"}


class _Scripted:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        return self.response


class _Exploding:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def call(self, system: str, user: str) -> dict[str, Any]:
        raise self.exc


# ---------- Both layers happy: penicillin recognized ----------


def test_happy_path_both_layers_populated(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "product": "penicillin",
                "process_family_hint": "fed-batch",
                "confidence": 0.92,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"},
                    {"paragraph_idx": 0, "span_text": "fed-batch mode"},
                    {"paragraph_idx": 0, "span_text": "penicillin production"},
                ],
                "rationale": "organism, product, family all named",
            },
            "registered": {
                "process_id": "penicillin_indpensim",
                "confidence": 0.92,
            },
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    # observed
    assert result.observed.provenance == IdentityProvenance.LLM_WHITELISTED
    assert result.observed.organism == "Penicillium chrysogenum"
    assert result.observed.product == "penicillin"
    assert result.observed.process_family_hint == "fed-batch"
    assert len(result.observed.evidence_locators) == 3
    # registered
    assert result.registered.provenance == IdentityProvenance.LLM_WHITELISTED
    assert result.registered.process_id == "penicillin_indpensim"


# ---------- The user requirement: yeast keeps observed even with no registry hit ----------


def test_yeast_case_preserves_observed_when_registered_fails(
    registry, yeast_blocks
):
    """Canonical yeast case: organism in prose, no registry entry -> observed
    populated, registered=UNKNOWN. This is the regression guard for the v4.1
    design pivot.
    """
    client = _Scripted(
        {
            "observed": {
                "organism": "Saccharomyces cerevisiae",
                "product": "ethanol",
                "process_family_hint": "batch",
                "confidence": 0.9,
                "evidence": [
                    {
                        "paragraph_idx": 0,
                        "span_text": "Saccharomyces cerevisiae",
                    },
                    {
                        "paragraph_idx": 0,
                        "span_text": "ethanol production",
                    },
                    {"paragraph_idx": 0, "span_text": "batch mode"},
                ],
            },
            "registered": {
                "process_id": None,
                "rationale": "no yeast process in registry",
            },
        }
    )
    result = IdentityExtractor(registry, client).extract(
        yeast_blocks, {"biomass_g_l", "ph"}
    )
    # The whole point: observed survives even though registered does not.
    assert result.observed.provenance == IdentityProvenance.LLM_WHITELISTED
    assert result.observed.organism == "Saccharomyces cerevisiae"
    assert result.observed.product == "ethanol"
    assert result.registered.provenance == IdentityProvenance.UNKNOWN
    assert result.registered.process_id is None
    assert "registry" in (result.registered.rationale or "").lower()


# ---------- Registered-layer failure modes ----------


def test_off_whitelist_process_id_keeps_observed(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "confidence": 0.8,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {"process_id": "made_up_process"},
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.organism == "Penicillium chrysogenum"
    assert result.registered.process_id is None
    assert "off-whitelist" in result.registered.rationale


def test_fingerprint_mismatch_keeps_observed(registry, penicillin_blocks):
    """LLM picks penicillin but dossier lacks the penicillin tracers."""
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "confidence": 0.9,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {"process_id": "penicillin_indpensim"},
        }
    )
    # paa_mg_l + nh3_mg_l absent from present_variables
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, {"biomass_g_l"}
    )
    # observed survives
    assert result.observed.organism == "Penicillium chrysogenum"
    assert result.observed.provenance == IdentityProvenance.LLM_WHITELISTED
    # registered fails
    assert result.registered.process_id is None
    assert "fingerprint mismatch" in result.registered.rationale


def test_null_process_id_keeps_observed(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "confidence": 0.7,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {"process_id": None, "rationale": "ambiguous"},
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.organism == "Penicillium chrysogenum"
    assert result.registered.process_id is None
    assert result.registered.rationale == "ambiguous"


# ---------- Observed-layer failure modes ----------


def test_unquoted_observed_field_is_nulled_others_survive(
    registry, penicillin_blocks, fingerprint_present
):
    """Field-level evidence check: unsupported fields drop, supported fields stay."""
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",  # supported
                "product": "magic_dust",  # NOT in any evidence span
                "confidence": 0.9,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {"process_id": "penicillin_indpensim"},
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.organism == "Penicillium chrysogenum"
    assert result.observed.product is None  # nulled by evidence check
    assert result.observed.provenance == IdentityProvenance.LLM_WHITELISTED


def test_evidence_cites_unknown_paragraph_rejected(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "evidence": [
                    {"paragraph_idx": 99, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {"process_id": "penicillin_indpensim"},
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    # No evidence survives -> observed UNKNOWN
    assert result.observed.provenance == IdentityProvenance.UNKNOWN


def test_no_observed_evidence_means_observed_unknown(
    registry, penicillin_blocks, fingerprint_present
):
    """LLM emits fields but no evidence at all -> observed=UNKNOWN."""
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "evidence": [],
            },
            "registered": {"process_id": "penicillin_indpensim"},
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.provenance == IdentityProvenance.UNKNOWN
    # registered still passes
    assert result.registered.process_id == "penicillin_indpensim"


# ---------- Confidence cap (both layers) ----------


def test_confidence_capped_in_both_layers(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "confidence": 0.99,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"}
                ],
            },
            "registered": {
                "process_id": "penicillin_indpensim",
                "confidence": 0.99,
            },
        }
    )
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.confidence == LLM_CONFIDENCE_CAP
    assert result.registered.confidence == LLM_CONFIDENCE_CAP


# ---------- Top-level failure modes (both layers UNKNOWN together) ----------


def test_empty_narrative_skips_llm_both_layers_unknown(
    registry, fingerprint_present
):
    client = _Scripted(
        {
            "observed": {"organism": "X"},
            "registered": {"process_id": "X"},
        }
    )
    result = IdentityExtractor(registry, client).extract([], fingerprint_present)
    assert result.observed.provenance == IdentityProvenance.UNKNOWN
    assert result.registered.provenance == IdentityProvenance.UNKNOWN
    assert client.calls == 0
    assert "no narrative" in result.observed.rationale.lower()


def test_llm_timeout_both_layers_unknown(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Exploding(TimeoutError("upstream timeout"))
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.provenance == IdentityProvenance.UNKNOWN
    assert result.registered.provenance == IdentityProvenance.UNKNOWN
    assert "TimeoutError" in result.observed.rationale


def test_llm_generic_error_both_layers_unknown(
    registry, penicillin_blocks, fingerprint_present
):
    client = _Exploding(RuntimeError("provider blew up"))
    result = IdentityExtractor(registry, client).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.provenance == IdentityProvenance.UNKNOWN
    assert result.registered.provenance == IdentityProvenance.UNKNOWN


def test_no_client_configured_both_layers_unknown(
    registry, penicillin_blocks, fingerprint_present
):
    result = IdentityExtractor(registry, client=None).extract(
        penicillin_blocks, fingerprint_present
    )
    assert result.observed.provenance == IdentityProvenance.UNKNOWN
    assert result.registered.provenance == IdentityProvenance.UNKNOWN
    assert "no LLM client" in result.observed.rationale
