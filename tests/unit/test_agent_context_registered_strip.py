"""Tests for AgentContext registered-rationale stripping.

When the identity extractor finds no registry entry that matches the
observed organism, it still writes a rationale string explaining what
it compared against. That string travels into every downstream agent's
prompt and biases their framing — we observed agents producing
"Penicillium vs S. cerevisiae" hypotheses on a yeast carotenoid PDF
solely because the registry's only entry is Penicillium.

The fix: strip `registered.rationale` when `registered.process_id is None`.
The flag (UNKNOWN_PROCESS) is the routing signal; the rationale is what
hijacks salience.

Invariant: when a real match exists (process_id is non-null), the
rationale is informative and stays.
"""

from __future__ import annotations

from fermdocs_characterize.agent_context import (
    _strip_unmatched_registered_rationale,
)


def _process(*, process_id, rationale, organism="S. cerevisiae"):
    return {
        "observed": {"organism": organism, "confidence": 0.85},
        "registered": {
            "process_id": process_id,
            "confidence": 0.0 if process_id is None else 0.9,
            "provenance": "unknown" if process_id is None else "llm_whitelisted",
            "rationale": rationale,
        },
    }


# ---------- unmatched cases: rationale must be stripped ----------


def test_strips_rationale_when_process_id_is_none():
    process = _process(process_id=None, rationale="S. cerevisiae does not match Penicillium chrysogenum")
    out = _strip_unmatched_registered_rationale(process)
    assert "rationale" not in out["registered"], (
        "rationale must be stripped when no registered process matched"
    )


def test_keeps_other_registered_fields_when_stripping():
    process = _process(process_id=None, rationale="some bias-inducing text")
    out = _strip_unmatched_registered_rationale(process)
    # process_id, confidence, provenance must survive
    assert out["registered"]["process_id"] is None
    assert out["registered"]["confidence"] == 0.0
    assert out["registered"]["provenance"] == "unknown"


def test_observed_block_untouched_when_stripping():
    """The whole point is to remove only the misleading comparison text;
    everything the LLM observed about the actual experiment must survive."""
    process = _process(process_id=None, rationale="x")
    out = _strip_unmatched_registered_rationale(process)
    assert out["observed"]["organism"] == "S. cerevisiae"
    assert out["observed"]["confidence"] == 0.85


# ---------- matched cases: rationale must be preserved ----------


def test_preserves_rationale_when_process_id_is_set():
    """When a real match exists (process_id non-null), the rationale is
    informative — it explains WHY the LLM picked that registry entry —
    and should remain."""
    process = _process(
        process_id="penicillin_indpensim",
        rationale="Organism matches P. chrysogenum entry exactly.",
    )
    out = _strip_unmatched_registered_rationale(process)
    assert out["registered"]["rationale"] == (
        "Organism matches P. chrysogenum entry exactly."
    )


# ---------- defensive cases ----------


def test_handles_missing_registered_block():
    process = {"observed": {"organism": "S. cerevisiae"}}
    out = _strip_unmatched_registered_rationale(process)
    assert out == process  # no change


def test_handles_non_dict_registered_block():
    """Defensive: a malformed dossier where registered is None or a
    string shouldn't crash."""
    process = {"observed": {}, "registered": None}
    out = _strip_unmatched_registered_rationale(process)
    assert out == process


def test_handles_non_dict_process():
    out = _strip_unmatched_registered_rationale("garbage")  # type: ignore[arg-type]
    assert out == "garbage"


def test_handles_empty_process_dict():
    out = _strip_unmatched_registered_rationale({})
    assert out == {}


def test_no_rationale_to_strip_is_no_op():
    """Already stripped, or never had one — no error."""
    process = {
        "observed": {"organism": "x"},
        "registered": {"process_id": None, "confidence": 0.0},
    }
    out = _strip_unmatched_registered_rationale(process)
    assert out["registered"] == {"process_id": None, "confidence": 0.0}


# ---------- integration: build_agent_context applies the strip ----------


def test_build_agent_context_strips_rationale_end_to_end():
    """The build_agent_context entry point must apply the strip — the
    helper being correct is necessary but not sufficient."""
    from fermdocs_characterize.agent_context import build_agent_context
    from datetime import datetime
    from uuid import UUID
    from fermdocs_characterize.schema import (
        CharacterizationOutput,
        Meta,
    )

    dossier = {
        "experiment": {
            "process": _process(
                process_id=None,
                rationale="LLM-fabricated comparison text that biases agents",
            ),
        },
        "ingestion_summary": {"schema_version": "2.0"},
        "golden_columns": {},
    }
    output = CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=UUID("11111111-1111-1111-1111-111111111111"),
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-TEST"],
        ),
        findings=[],
    )
    ctx = build_agent_context(dossier, output)
    assert "rationale" not in ctx.process["registered"], (
        "build_agent_context must apply the strip — the bias-inducing "
        "rationale leaked through"
    )
    assert ctx.process["observed"]["organism"] == "S. cerevisiae"


def test_build_agent_context_preserves_rationale_on_real_match():
    """Sanity: when there IS a registered match, the rationale stays
    (it's the model's explanation of why it picked that entry, which is
    legitimately informative for downstream agents)."""
    from fermdocs_characterize.agent_context import build_agent_context
    from datetime import datetime
    from uuid import UUID
    from fermdocs_characterize.schema import (
        CharacterizationOutput,
        Meta,
    )

    dossier = {
        "experiment": {
            "process": _process(
                process_id="penicillin_indpensim",
                rationale="Matches P. chrysogenum + paa_mg_l fingerprint.",
            ),
        },
        "ingestion_summary": {"schema_version": "2.0"},
        "golden_columns": {},
    }
    output = CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=UUID("11111111-1111-1111-1111-111111111111"),
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-TEST"],
        ),
        findings=[],
    )
    ctx = build_agent_context(dossier, output)
    assert ctx.process["registered"]["rationale"] == (
        "Matches P. chrysogenum + paa_mg_l fingerprint."
    )
