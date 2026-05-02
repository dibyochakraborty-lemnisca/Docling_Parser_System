"""Integration tests for build_dossier with the identity extension.

These tests mock the repository to keep them fast and DB-free. They verify
the priority chain (manifest > LLM extractor > UNKNOWN) and the two-layer
shape end to end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fermdocs.domain.models import IdentityProvenance
from fermdocs.dossier import build_dossier, load_process_manifest


class _FakeExperimentRow:
    def __init__(self, exp_id: str) -> None:
        self.experiment_id = exp_id
        self.name = None
        self.uploaded_by = None
        self.created_at = datetime.now(timezone.utc)
        self.status = "ingested"


class _FakeResidualRow:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.residual_id = "R-1"
        self.file_id = "F-1"
        self.extractor_version = "v0.1.0"
        self.payload = payload


class _FakeRepo:
    """Minimum surface build_dossier needs."""

    def __init__(
        self,
        experiment_id: str,
        residuals: list[_FakeResidualRow] | None = None,
    ) -> None:
        self._exp = _FakeExperimentRow(experiment_id)
        self._residuals = residuals or []

    def fetch_experiment(self, experiment_id: str):
        return self._exp

    def fetch_files(self, experiment_id: str):
        return []

    def fetch_active_observations(self, experiment_id: str):
        return []

    def fetch_residuals(self, experiment_id: str):
        return self._residuals

    def row_to_observation(self, row):
        raise NotImplementedError


class _Scripted:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        return self.response


def _residual_with_narrative(text: str) -> _FakeResidualRow:
    return _FakeResidualRow(
        {
            "narrative": [
                {
                    "text": text,
                    "type": "paragraph",
                    "locator": {
                        "paragraph_idx": 0,
                        "page": 1,
                        "file_id": "F-1",
                    },
                }
            ]
        }
    )


# ---------- Manifest path ----------


def test_manifest_path_pins_both_layers_to_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "process_id: penicillin_indpensim\n"
        "organism: Penicillium chrysogenum\n"
        "product: penicillin\n"
        "process_family: fed-batch\n"
        "scale:\n"
        "  volume_l: 58000.0\n"
        "  vessel_type: stirred-tank\n"
        "confidence: 1.0\n"
    )
    repo = _FakeRepo(
        "EXP-1", residuals=[_residual_with_narrative("some narrative text")]
    )
    client = _Scripted({"observed": {}, "registered": {}})

    dossier = build_dossier(
        "EXP-1", repo, manifest_path=manifest, identity_llm_client=client
    )
    process = dossier["experiment"]["process"]
    assert process["observed"]["provenance"] == IdentityProvenance.MANIFEST.value
    assert process["observed"]["organism"] == "Penicillium chrysogenum"
    assert process["observed"]["product"] == "penicillin"
    assert process["observed"]["scale"]["volume_l"] == 58000.0
    assert process["registered"]["provenance"] == IdentityProvenance.MANIFEST.value
    assert process["registered"]["process_id"] == "penicillin_indpensim"
    # LLM never called
    assert client.calls == 0


def test_manifest_can_omit_process_id(tmp_path: Path):
    """Operator knows organism/product but not the registry id -> observed
    populated, registered.process_id null but provenance still MANIFEST."""
    manifest = tmp_path / "yeast_manifest.yaml"
    manifest.write_text(
        "organism: Saccharomyces cerevisiae\n"
        "product: ethanol\n"
        "process_family: batch\n"
    )
    identity = load_process_manifest(manifest)
    assert identity.observed.organism == "Saccharomyces cerevisiae"
    assert identity.observed.provenance == IdentityProvenance.MANIFEST
    assert identity.registered.process_id is None
    assert identity.registered.provenance == IdentityProvenance.MANIFEST


def test_load_manifest_rejects_non_mapping(tmp_path: Path):
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("- not - a - mapping\n")
    with pytest.raises(ValueError):
        load_process_manifest(manifest)


# ---------- Extractor success and fallbacks ----------


def test_extractor_yeast_observed_populated_registered_unknown(monkeypatch):
    """The user requirement: yeast experiments preserve organism + product
    even when the registry has no match."""
    from fermdocs import dossier as dossier_module

    monkeypatch.setattr(
        dossier_module,
        "_present_variables",
        lambda by_column: {"biomass_g_l", "ph"},  # no penicillin tracers
    )
    repo = _FakeRepo(
        "YEAST-1",
        residuals=[
            _residual_with_narrative(
                "Saccharomyces cerevisiae was used for ethanol production in batch mode."
            )
        ],
    )
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
                    {"paragraph_idx": 0, "span_text": "ethanol production"},
                    {"paragraph_idx": 0, "span_text": "batch mode"},
                ],
            },
            "registered": {
                "process_id": None,
                "rationale": "no yeast process in registry",
            },
        }
    )
    dossier = build_dossier("YEAST-1", repo, identity_llm_client=client)
    process = dossier["experiment"]["process"]
    # observed survives
    assert (
        process["observed"]["provenance"]
        == IdentityProvenance.LLM_WHITELISTED.value
    )
    assert process["observed"]["organism"] == "Saccharomyces cerevisiae"
    assert process["observed"]["product"] == "ethanol"
    # registered does not
    assert process["registered"]["provenance"] == IdentityProvenance.UNKNOWN.value
    assert process["registered"]["process_id"] is None


def test_extractor_penicillin_both_layers_populated(monkeypatch):
    from fermdocs import dossier as dossier_module

    monkeypatch.setattr(
        dossier_module,
        "_present_variables",
        lambda by_column: {"paa_mg_l", "nh3_mg_l", "biomass_g_l"},
    )
    repo = _FakeRepo(
        "PEN-1",
        residuals=[
            _residual_with_narrative(
                "Penicillium chrysogenum was cultivated for penicillin production with PAA feed."
            )
        ],
    )
    client = _Scripted(
        {
            "observed": {
                "organism": "Penicillium chrysogenum",
                "product": "penicillin",
                "confidence": 0.9,
                "evidence": [
                    {"paragraph_idx": 0, "span_text": "Penicillium chrysogenum"},
                    {"paragraph_idx": 0, "span_text": "penicillin production"},
                ],
            },
            "registered": {
                "process_id": "penicillin_indpensim",
                "confidence": 0.9,
            },
        }
    )
    dossier = build_dossier("PEN-1", repo, identity_llm_client=client)
    process = dossier["experiment"]["process"]
    assert (
        process["observed"]["provenance"]
        == IdentityProvenance.LLM_WHITELISTED.value
    )
    assert (
        process["registered"]["provenance"]
        == IdentityProvenance.LLM_WHITELISTED.value
    )
    assert process["registered"]["process_id"] == "penicillin_indpensim"


def test_extractor_no_client_falls_through_to_unknown():
    repo = _FakeRepo(
        "EXP-NO-CLIENT",
        residuals=[_residual_with_narrative("some prose")],
    )
    dossier = build_dossier("EXP-NO-CLIENT", repo)
    process = dossier["experiment"]["process"]
    assert process["observed"]["provenance"] == IdentityProvenance.UNKNOWN.value
    assert process["registered"]["provenance"] == IdentityProvenance.UNKNOWN.value


def test_dossier_schema_version_bumped_to_1_1():
    repo = _FakeRepo("EXP-VERSION")
    dossier = build_dossier("EXP-VERSION", repo)
    assert dossier["dossier_schema_version"] == "1.1"
