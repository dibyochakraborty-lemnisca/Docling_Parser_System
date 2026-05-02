"""Tests for the identity + diagnosis client factories.

These cover the factory dispatch logic only — they do NOT make live API calls
to Gemini or Anthropic. Instantiating a client is allowed (it's just storing
config); the actual `.call(...)` would talk to the network and is excluded.
"""

from __future__ import annotations

import os

import pytest

from fermdocs.mapping.factory import (
    UnknownProviderError,
    build_identity_client,
)
from fermdocs.mapping.anthropic_identity_client import AnthropicIdentityClient
from fermdocs.mapping.gemini_identity_client import GeminiIdentityClient
from fermdocs_diagnose.llm_clients import (
    AnthropicDiagnosisClient,
    GeminiDiagnosisClient,
    build_diagnosis_client,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test sees a deterministic default."""
    for key in (
        "FERMDOCS_IDENTITY_PROVIDER",
        "FERMDOCS_DIAGNOSIS_PROVIDER",
        "FERMDOCS_MAPPER_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------- identity factory ----------


def test_identity_factory_default_is_gemini():
    client = build_identity_client()
    assert isinstance(client, GeminiIdentityClient)


def test_identity_factory_explicit_anthropic():
    client = build_identity_client("anthropic")
    assert isinstance(client, AnthropicIdentityClient)


def test_identity_factory_explicit_gemini():
    client = build_identity_client("gemini")
    assert isinstance(client, GeminiIdentityClient)


def test_identity_factory_none_returns_none():
    """'none' / 'fake' short-circuit so offline runs leave identity UNKNOWN."""
    assert build_identity_client("none") is None
    assert build_identity_client("fake") is None


def test_identity_factory_case_insensitive():
    assert isinstance(build_identity_client("Gemini"), GeminiIdentityClient)
    assert isinstance(build_identity_client("ANTHROPIC"), AnthropicIdentityClient)


def test_identity_factory_env_var_default(monkeypatch):
    monkeypatch.setenv("FERMDOCS_IDENTITY_PROVIDER", "anthropic")
    assert isinstance(build_identity_client(), AnthropicIdentityClient)


def test_identity_factory_env_chain_falls_back_to_mapper(monkeypatch):
    """If FERMDOCS_IDENTITY_PROVIDER is unset, MAPPER_PROVIDER is consulted."""
    monkeypatch.setenv("FERMDOCS_MAPPER_PROVIDER", "anthropic")
    assert isinstance(build_identity_client(), AnthropicIdentityClient)


def test_identity_factory_unknown_provider_raises():
    with pytest.raises(UnknownProviderError):
        build_identity_client("openai")


# ---------- diagnosis factory ----------


def test_diagnosis_factory_default_is_gemini():
    client = build_diagnosis_client()
    assert isinstance(client, GeminiDiagnosisClient)


def test_diagnosis_factory_explicit_anthropic():
    client = build_diagnosis_client("anthropic")
    assert isinstance(client, AnthropicDiagnosisClient)


def test_diagnosis_factory_none_returns_none():
    assert build_diagnosis_client("none") is None
    assert build_diagnosis_client("fake") is None


def test_diagnosis_factory_env_var_default(monkeypatch):
    monkeypatch.setenv("FERMDOCS_DIAGNOSIS_PROVIDER", "anthropic")
    assert isinstance(build_diagnosis_client(), AnthropicDiagnosisClient)


def test_diagnosis_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_diagnosis_client("openai")


# ---------- model resolution ----------


def test_gemini_identity_client_picks_up_model_env(monkeypatch):
    monkeypatch.setenv("FERMDOCS_IDENTITY_MODEL", "gemini-3-pro-test")
    c = GeminiIdentityClient()
    assert c._model == "gemini-3-pro-test"


def test_anthropic_diagnosis_client_picks_up_model_env(monkeypatch):
    monkeypatch.setenv("FERMDOCS_DIAGNOSIS_MODEL", "claude-test-model")
    c = AnthropicDiagnosisClient()
    assert c._model == "claude-test-model"


def test_gemini_identity_client_falls_back_to_gemini_model_env(monkeypatch):
    """When IDENTITY_MODEL not set, FERMDOCS_GEMINI_MODEL applies."""
    monkeypatch.delenv("FERMDOCS_IDENTITY_MODEL", raising=False)
    monkeypatch.setenv("FERMDOCS_GEMINI_MODEL", "gemini-3-flash-test")
    c = GeminiIdentityClient()
    assert c._model == "gemini-3-flash-test"
