"""Diagnosis-agent eval runner.

Plan ref: plans/2026-05-02-diagnosis-agent.md §11.

Each fixture lives in its own subdirectory and consists of:
  - scripted_llm_response.json: what a "good" LLM would emit for this dossier
  - expected_claims.yaml: scoring spec
  - (optional) reuses_upstream: name of evals/characterize/fixtures/<name>
    whose dossier and pipeline output to reuse.

The runner:
  1. Loads the upstream characterize fixture's dossier.json.
  2. Runs CharacterizationPipeline to produce a CharacterizationOutput.
  3. Wires a scripted client that returns scripted_llm_response.json.
  4. Calls DiagnosisAgent.diagnose(...).
  5. Scores the resulting DiagnosisOutput against expected_claims.yaml.

Hard pass/fail axes:
  - citation_integrity: every cited finding must resolve
  - honesty_under_unknown_flags: no surviving process_priors when flagged
  - expected_provenance_downgrades: every flagged claim must carry
    provenance_downgraded=True

Soft (warn-only):
  - forbidden_phrases
  - claim recall (logged for visibility, doesn't fail)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from fermdocs_characterize.agent_context import build_agent_context
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_diagnose.agent import DiagnosisAgent
from fermdocs_diagnose.schema import ConfidenceBasis, DiagnosisOutput

EVALS_DIR = Path(__file__).resolve().parent
CHARACTERIZE_FIXTURES = (
    EVALS_DIR.parent / "characterize" / "fixtures"
)

FIXTURES = ["01_boundary", "02_missing_data", "03_multi_run", "04_unknown_everything"]


class _ScriptedClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls = 0

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.calls += 1
        return self._response


def _load_fixture(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Returns (dossier, scripted_response, expected)."""
    fixture_dir = EVALS_DIR / name
    expected = yaml.safe_load((fixture_dir / "expected_claims.yaml").read_text())
    upstream_name = expected.get("reuses_upstream", name)
    dossier = json.loads(
        (CHARACTERIZE_FIXTURES / upstream_name / "dossier.json").read_text()
    )
    scripted = json.loads((fixture_dir / "scripted_llm_response.json").read_text())
    return dossier, scripted, expected


def _run_diagnosis(dossier: dict[str, Any], scripted: dict[str, Any]) -> DiagnosisOutput:
    upstream = CharacterizationPipeline(validate=False).run(
        dossier,
        characterization_id=uuid.UUID(int=0),
        generation_timestamp=datetime(2026, 1, 1),
    )
    agent = DiagnosisAgent(client=_ScriptedClient(scripted))
    return agent.diagnose(
        dossier,
        upstream,
        diagnosis_id=uuid.UUID(int=99),
        generation_timestamp=datetime(2026, 5, 2),
    )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_no_meta_error(fixture_name: str):
    """Agent produces a usable output (not a meta.error escape hatch)."""
    dossier, scripted, _ = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    assert out.meta.error is None, (
        f"{fixture_name}: agent produced meta.error={out.meta.error!r}"
    )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_citation_integrity(fixture_name: str):
    """Every surviving claim cites a finding that exists upstream."""
    dossier, scripted, _ = _load_fixture(fixture_name)
    upstream = CharacterizationPipeline(validate=False).run(
        dossier,
        characterization_id=uuid.UUID(int=0),
        generation_timestamp=datetime(2026, 1, 1),
    )
    out = _run_diagnosis(dossier, scripted)
    finding_ids = {f.finding_id for f in upstream.findings}
    for claim in (*out.failures, *out.analysis):
        for fid in claim.cited_finding_ids:
            assert fid in finding_ids, (
                f"{fixture_name}: {claim.claim_id} cites unknown finding {fid!r}"
            )
    for q in out.open_questions:
        for fid in q.cited_finding_ids:
            assert fid in finding_ids, (
                f"{fixture_name}: {q.question_id} cites unknown finding {fid!r}"
            )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_honesty_under_unknown_flags(fixture_name: str):
    """When UNKNOWN_PROCESS or UNKNOWN_ORGANISM is in flags, no surviving
    claim may carry confidence_basis=process_priors. The validator should
    have downgraded them all.
    """
    dossier, scripted, expected = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    upstream = CharacterizationPipeline(validate=False).run(
        dossier,
        characterization_id=uuid.UUID(int=0),
        generation_timestamp=datetime(2026, 1, 1),
    )
    ctx = build_agent_context(dossier, upstream)
    flag_values = {f.value for f in ctx.flags}
    if "unknown_process" in flag_values or "unknown_organism" in flag_values:
        for claim in (*out.failures, *out.trends, *out.analysis):
            assert claim.confidence_basis != ConfidenceBasis.PROCESS_PRIORS, (
                f"{fixture_name}: {claim.claim_id} survived as process_priors"
                f" under flags {sorted(flag_values)}"
            )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_expected_failures_match(fixture_name: str):
    """Each expected_failure must have a matching emitted failure."""
    dossier, scripted, expected = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    for ef in expected.get("expected_failures") or []:
        keywords = [k.lower() for k in ef["summary_keywords"]]
        must_cite = set(ef["must_cite_findings"])
        matches = [
            f
            for f in out.failures
            if all(k in f.summary.lower() for k in keywords)
            and must_cite.issubset(set(f.cited_finding_ids))
        ]
        assert matches, (
            f"{fixture_name}: no failure matched keywords={keywords} citing {must_cite}"
        )
        if "min_confidence" in ef:
            assert any(m.confidence >= ef["min_confidence"] for m in matches), (
                f"{fixture_name}: matching failure below min_confidence"
            )
        if "must_have_domain_tags" in ef:
            for tag in ef["must_have_domain_tags"]:
                assert any(tag in m.domain_tags for m in matches), (
                    f"{fixture_name}: matching failure missing domain_tag {tag!r}"
                )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_expected_open_questions_match(fixture_name: str):
    dossier, scripted, expected = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    for eq in expected.get("expected_open_questions") or []:
        keywords = [k.lower() for k in eq["keywords"]]
        must_cite = set(eq["must_cite_findings"])
        matches = [
            q
            for q in out.open_questions
            if any(k in q.question.lower() for k in keywords)
            and must_cite.issubset(set(q.cited_finding_ids))
        ]
        assert matches, (
            f"{fixture_name}: no open question matched keywords={keywords}"
            f" citing {must_cite}"
        )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_min_open_questions(fixture_name: str):
    _, _, expected = _load_fixture(fixture_name)
    minimum = expected.get("min_open_questions")
    if minimum is None:
        pytest.skip("no min_open_questions specified")
    dossier, scripted, _ = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    assert len(out.open_questions) >= minimum, (
        f"{fixture_name}: {len(out.open_questions)} questions, expected ≥{minimum}"
    )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_expected_provenance_downgrades(fixture_name: str):
    _, _, expected = _load_fixture(fixture_name)
    spec = expected.get("expected_provenance_downgrades") or []
    if not spec:
        pytest.skip("no expected_provenance_downgrades")
    dossier, scripted, _ = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    by_kind = {
        "failure": out.failures,
        "trend": out.trends,
        "analysis": out.analysis,
    }
    for entry in spec:
        kind = entry["claim_kind"]
        cited = set(entry["cited_findings"])
        candidates = [
            c for c in by_kind[kind] if cited.issubset(set(c.cited_finding_ids))
        ]
        assert candidates, (
            f"{fixture_name}: no {kind} claim citing {cited} found at all"
        )
        downgraded = [c for c in candidates if c.provenance_downgraded]
        assert downgraded, (
            f"{fixture_name}: expected {kind} citing {cited} to carry"
            f" provenance_downgraded=True, none did"
        )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_forbidden_phrases_soft_check(fixture_name: str, caplog):
    """Forbidden causal phrases are warn-only. Test asserts the warning fires
    if any are present, doesn't fail the eval.
    """
    dossier, scripted, expected = _load_fixture(fixture_name)
    forbidden = [p.lower() for p in (expected.get("forbidden_phrases") or [])]
    if not forbidden:
        pytest.skip("no forbidden_phrases configured")
    with caplog.at_level(logging.WARNING):
        out = _run_diagnosis(dossier, scripted)
    summaries = " ".join(
        c.summary.lower()
        for c in (*out.failures, *out.trends, *out.analysis)
    )
    has_forbidden = any(p in summaries for p in forbidden)
    if has_forbidden:
        assert any("causal phrasing" in r.message for r in caplog.records), (
            f"{fixture_name}: forbidden phrase present but no warning logged"
        )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_expected_analysis_match(fixture_name: str):
    _, _, expected = _load_fixture(fixture_name)
    spec = expected.get("expected_analysis") or []
    if not spec:
        pytest.skip("no expected_analysis")
    dossier, scripted, _ = _load_fixture(fixture_name)
    out = _run_diagnosis(dossier, scripted)
    for ea in spec:
        keywords = [k.lower() for k in ea["summary_keywords"]]
        must_cite = set(ea["must_cite_findings"])
        matches = [
            a
            for a in out.analysis
            if all(k in a.summary.lower() for k in keywords)
            and must_cite.issubset(set(a.cited_finding_ids))
        ]
        assert matches, (
            f"{fixture_name}: no analysis matched keywords={keywords} citing"
            f" {must_cite}"
        )
        if "must_have_kind" in ea:
            assert any(m.kind == ea["must_have_kind"] for m in matches), (
                f"{fixture_name}: analysis missing kind={ea['must_have_kind']}"
            )
        if "must_have_confidence_basis" in ea:
            wanted = ConfidenceBasis(ea["must_have_confidence_basis"])
            assert any(m.confidence_basis == wanted for m in matches), (
                f"{fixture_name}: analysis missing confidence_basis={wanted}"
            )
