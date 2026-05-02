"""Tests for the diagnosis ReAct agent.

Scripted LLM clients drive the loop deterministically. The agent should:
  - return error-output when no client is configured
  - return error-output when the client raises
  - dispatch tool calls and feed results back into the loop
  - terminate on `action: emit` with a typed DiagnosisOutput
  - assign claim_ids deterministically by position
  - apply soft enforcement (provenance downgrade) via validate_diagnosis
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Trajectory,
)
from fermdocs_characterize.specs import DictSpecsProvider, Spec
from fermdocs_diagnose.agent import DiagnosisAgent
from fermdocs_diagnose.schema import ConfidenceBasis


CHAR_ID = uuid.UUID(int=42)
DIAG_ID = uuid.UUID(int=99)


@pytest.fixture
def upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"],
        ),
        findings=[
            Finding(
                finding_id=f"{CHAR_ID}:F-0001",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MAJOR,
                summary="biomass below nominal",
                confidence=0.8,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=3, n_independent_runs=1),
                evidence_observation_ids=["O-1", "O-2", "O-3"],
                variables_involved=["biomass_g_l"],
                statistics={"sigma": 2.5, "observed": 7.5, "nominal": 10.0},
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-1",
                variable="biomass_g_l",
                time_grid=[0.0, 1.0],
                values=[1.0, 2.0],
                imputation_flags=[False, False],
                source_observation_ids=["O-1"],
                unit="g/L",
                quality=1.0,
                data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
            )
        ],
    )


@pytest.fixture
def dossier() -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_id": "EXP-X",
            "process": {
                "observed": {
                    "organism": "Penicillium chrysogenum",
                    "product": "penicillin",
                    "provenance": "llm_whitelisted",
                },
                "registered": {
                    "process_id": "penicillin_indpensim",
                    "provenance": "llm_whitelisted",
                },
            },
        },
        "ingestion_summary": {
            "schema_version": "2.0",
            "stale_schema_versions": [],
            "golden_coverage_percent": 80,
        },
        "golden_columns": {},
    }


@pytest.fixture
def specs() -> DictSpecsProvider:
    return DictSpecsProvider(
        {"biomass_g_l": Spec(nominal=10.0, std_dev=1.0, unit="g/L", provenance="schema")}
    )


class _ScriptedClient:
    """Returns a queue of responses, one per call."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.calls += 1
        if not self._responses:
            raise AssertionError("scripted client exhausted")
        return self._responses.pop(0)


class _ExplodingClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.calls += 1
        raise self.exc


def _emit(failures=None, trends=None, analysis=None, open_questions=None, narrative=None):
    return {
        "action": "emit",
        "failures": failures or [],
        "trends": trends or [],
        "analysis": analysis or [],
        "open_questions": open_questions or [],
        "narrative": narrative,
    }


def _diagnose(agent, dossier, upstream, specs):
    return agent.diagnose(
        dossier,
        upstream,
        specs_provider=specs,
        diagnosis_id=DIAG_ID,
        generation_timestamp=datetime(2026, 5, 2),
    )


# ---------- Failure paths ----------


def test_no_client_returns_error_output(dossier, upstream, specs):
    agent = DiagnosisAgent(client=None)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error == "no_llm_client_configured"
    assert out.failures == []


def test_llm_call_raises_returns_error_output(dossier, upstream, specs):
    agent = DiagnosisAgent(client=_ExplodingClient(RuntimeError("boom")))
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error == "llm_call_failed:RuntimeError"
    assert out.failures == []


def test_step_budget_exhausted_returns_error_output(dossier, upstream, specs):
    """Client never emits; loop runs until budget is hit."""
    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "get_finding", "args": {"finding_id": f"{CHAR_ID}:F-0001"}}
            for _ in range(10)
        ]
    )
    agent = DiagnosisAgent(client=client, max_steps=3)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error == "step_budget_exhausted"
    assert client.calls == 3


# ---------- Tool dispatch ----------


def test_tool_call_dispatched_then_emit(dossier, upstream, specs):
    """First step requests get_finding, second step emits."""
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_finding",
                "args": {"finding_id": f"{CHAR_ID}:F-0001"},
            },
            _emit(
                failures=[
                    {
                        "summary": "biomass plateau between 0h and 1h",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "affected_variables": ["biomass_g_l"],
                        "confidence": 0.7,
                        "confidence_basis": "schema_only",
                        "domain_tags": ["growth"],
                        "severity": "major",
                    }
                ]
            ),
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error is None
    assert len(out.failures) == 1
    assert out.failures[0].claim_id == "D-F-0001"
    assert client.calls == 2


def test_unknown_tool_returns_hint(dossier, upstream, specs):
    """Bad tool name doesn't crash the loop; agent gets a hint and can recover."""
    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "lookup_god", "args": {}},
            _emit(),
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error is None
    assert client.calls == 2


def test_get_trajectory_dispatch(dossier, upstream, specs):
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_trajectory",
                "args": {"run_id": "RUN-1", "variable": "biomass_g_l"},
            },
            _emit(),
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error is None


def test_get_spec_dispatch(dossier, upstream, specs):
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_spec",
                "args": {"variable": "biomass_g_l"},
            },
            _emit(),
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error is None


# ---------- Emit shapes ----------


def test_emit_with_all_claim_kinds(dossier, upstream, specs):
    client = _ScriptedClient(
        [
            _emit(
                failures=[
                    {
                        "summary": "biomass plateau",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "confidence": 0.7,
                        "confidence_basis": "schema_only",
                        "severity": "major",
                    }
                ],
                trends=[
                    {
                        "summary": "biomass plateau over 0-1h",
                        "cited_trajectories": [
                            {"run_id": "RUN-1", "variable": "biomass_g_l"}
                        ],
                        "confidence": 0.6,
                        "confidence_basis": "schema_only",
                        "direction": "plateau",
                    }
                ],
                analysis=[
                    {
                        "summary": "early plateau aligns with stationary phase",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "confidence": 0.6,
                        "confidence_basis": "schema_only",
                        "kind": "phase_characterization",
                    }
                ],
                open_questions=[
                    {
                        "question": "Was sampling frequency reduced after 30 min?",
                        "why_it_matters": "could explain the plateau as artifact",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "answer_format_hint": "yes_no",
                    }
                ],
                narrative="Run shows an early biomass plateau cited from F-0001.",
            )
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.error is None
    assert out.failures[0].claim_id == "D-F-0001"
    assert out.trends[0].claim_id == "D-T-0001"
    assert out.analysis[0].claim_id == "D-A-0001"
    assert out.open_questions[0].question_id == "D-Q-0001"
    assert out.narrative is not None


def test_deterministic_claim_id_assignment(dossier, upstream, specs):
    """Same emit payload → same IDs every time. The runtime owns IDs, not the LLM."""
    payload = _emit(
        failures=[
            {
                "summary": f"failure {i}",
                "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                "confidence": 0.5,
                "confidence_basis": "schema_only",
                "severity": "minor",
            }
            for i in range(3)
        ]
    )
    client1 = _ScriptedClient([payload])
    client2 = _ScriptedClient([payload])
    out1 = _diagnose(DiagnosisAgent(client=client1), dossier, upstream, specs)
    out2 = _diagnose(DiagnosisAgent(client=client2), dossier, upstream, specs)
    assert [c.claim_id for c in out1.failures] == ["D-F-0001", "D-F-0002", "D-F-0003"]
    assert [c.claim_id for c in out1.failures] == [c.claim_id for c in out2.failures]


# ---------- Validator integration ----------


def test_unknown_citation_dropped_after_emit(dossier, upstream, specs):
    """The agent emits a claim with a bogus citation; validator drops it."""
    client = _ScriptedClient(
        [
            _emit(
                failures=[
                    {
                        "summary": "real plateau",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "confidence": 0.6,
                        "confidence_basis": "schema_only",
                        "severity": "minor",
                    },
                    {
                        "summary": "fake claim",
                        "cited_finding_ids": [f"{CHAR_ID}:F-9999"],
                        "confidence": 0.6,
                        "confidence_basis": "schema_only",
                        "severity": "minor",
                    },
                ]
            )
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert len(out.failures) == 1
    assert out.failures[0].summary == "real plateau"


def test_provenance_downgrade_under_unknown_process(upstream, specs):
    """Yeast-style dossier where registered is UNKNOWN. Even if the LLM emits
    process_priors, the validator must downgrade.
    """
    yeast_dossier = {
        "experiment": {
            "experiment_id": "EXP-Y",
            "process": {
                "observed": {
                    "organism": "Saccharomyces cerevisiae",
                    "provenance": "llm_whitelisted",
                },
                "registered": {"process_id": None, "provenance": "unknown"},
            },
        },
        "ingestion_summary": {
            "schema_version": "2.0",
            "stale_schema_versions": [],
            "golden_coverage_percent": 80,
        },
        "golden_columns": {},
    }
    client = _ScriptedClient(
        [
            _emit(
                failures=[
                    {
                        "summary": "biomass plateau observed",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "confidence": 0.7,
                        "confidence_basis": "process_priors",
                        "severity": "major",
                    }
                ]
            )
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = agent.diagnose(
        yeast_dossier,
        upstream,
        specs_provider=specs,
        diagnosis_id=DIAG_ID,
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert out.failures[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert out.failures[0].provenance_downgraded is True


# ---------- Confidence clamping ----------


def test_emit_with_oversized_confidence_clamped(dossier, upstream, specs):
    """LLM returns confidence > 0.85; runtime clamps so schema validator passes."""
    client = _ScriptedClient(
        [
            _emit(
                failures=[
                    {
                        "summary": "biomass plateau",
                        "cited_finding_ids": [f"{CHAR_ID}:F-0001"],
                        "confidence": 0.99,
                        "confidence_basis": "schema_only",
                        "severity": "minor",
                    }
                ]
            )
        ]
    )
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.failures[0].confidence <= 0.85


# ---------- Meta wiring ----------


def test_meta_supersedes_links_to_characterization(dossier, upstream, specs):
    client = _ScriptedClient([_emit()])
    agent = DiagnosisAgent(client=client)
    out = _diagnose(agent, dossier, upstream, specs)
    assert out.meta.supersedes_characterization_id == CHAR_ID
    assert out.meta.diagnosis_id == DIAG_ID
    assert out.meta.schema_version == "1.0"
    assert out.meta.provider == "anthropic"
