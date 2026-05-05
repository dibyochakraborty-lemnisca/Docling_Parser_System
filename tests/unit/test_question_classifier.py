"""Tests for the question classifier (PR-A commit 2).

Hits the validation + fallback paths with a mock LLM client; the
production GeminiQuestionClassifierClient is exercised only through
its constructor (live calls are out of scope for unit tests).
"""

from __future__ import annotations

from typing import Any

import pytest

from fermdocs_hypothesis.question_classifier import (
    GeminiQuestionClassifierClient,
    classify_user_question,
)


class _ScriptedClient:
    """Mock QuestionClassifierLLMClient: returns a canned dict."""

    def __init__(self, payload: dict[str, Any] | Exception):
        self._payload = payload
        self.calls: list[tuple[str, str]] = []

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# ---------- happy path ----------


def test_classifier_returns_valid_shape_and_hints() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": ["RUN-0002"],
        "affected_variables": ["biomass_g_l"],
    })
    q = classify_user_question(
        text="Why did RUN-0002's biomass plateau early?",
        available_run_ids=["RUN-0001", "RUN-0002"],
        available_variables=["biomass_g_l", "wcw_g_l"],
        client=client,
    )
    assert q.shape == "scoping"
    assert q.affected_runs == ["RUN-0002"]
    assert q.affected_variables == ["biomass_g_l"]
    assert q.text == "Why did RUN-0002's biomass plateau early?"
    assert q.raised_by == "user"


def test_classifier_passes_text_through_verbatim() -> None:
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": [],
    })
    text = "What's going on with this experiment?"
    q = classify_user_question(
        text=text,
        available_run_ids=[],
        available_variables=[],
        client=client,
    )
    assert q.text == text


def test_classifier_respects_raised_by() -> None:
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="follow-up question",
        available_run_ids=[],
        available_variables=[],
        client=client,
        raised_by="user_followup",
    )
    assert q.raised_by == "user_followup"


# ---------- shape validation ----------


def test_classifier_falls_back_to_open_on_unknown_shape() -> None:
    client = _ScriptedClient({
        "shape": "speculative",  # not in VALID_SHAPES
        "affected_runs": [],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=[],
        client=client,
    )
    assert q.shape == "open"


def test_classifier_handles_missing_shape_field() -> None:
    client = _ScriptedClient({
        "affected_runs": [],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=[],
        client=client,
    )
    assert q.shape == "open"


def test_classifier_handles_uppercase_shape() -> None:
    client = _ScriptedClient({
        "shape": "SCOPING",
        "affected_runs": [],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?", available_run_ids=[], available_variables=[], client=client
    )
    assert q.shape == "scoping"


# ---------- run validation ----------


def test_classifier_drops_unknown_run_ids() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": ["RUN-0002", "RUN-9999"],  # 9999 is hallucinated
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=["RUN-0001", "RUN-0002"],
        available_variables=[],
        client=client,
    )
    assert q.affected_runs == ["RUN-0002"]


def test_classifier_canonicalizes_case_for_runs() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": ["run-0002", "RuN-0001"],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=["RUN-0001", "RUN-0002"],
        available_variables=[],
        client=client,
    )
    assert sorted(q.affected_runs) == ["RUN-0001", "RUN-0002"]


def test_classifier_dedups_runs() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": ["RUN-0002", "RUN-0002", "run-0002"],
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=["RUN-0001", "RUN-0002"],
        available_variables=[],
        client=client,
    )
    assert q.affected_runs == ["RUN-0002"]


def test_classifier_drops_non_string_runs() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": ["RUN-0001", 42, None],  # mixed garbage
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=["RUN-0001"],
        available_variables=[],
        client=client,
    )
    assert q.affected_runs == ["RUN-0001"]


# ---------- variable validation ----------


def test_classifier_keeps_exact_variable_match() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": [],
        "affected_variables": ["biomass_g_l"],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=["biomass_g_l", "wcw_g_l"],
        client=client,
    )
    assert q.affected_variables == ["biomass_g_l"]


def test_classifier_substring_matches_variables() -> None:
    """LLM emits 'biomass'; bundle has 'biomass_g_l' — should match."""
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": [],
        "affected_variables": ["biomass"],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=["biomass_g_l", "wcw_g_l"],
        client=client,
    )
    assert q.affected_variables == ["biomass_g_l"]


def test_classifier_drops_unknown_variables() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": [],
        "affected_variables": ["biomass_g_l", "made_up_var"],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=["biomass_g_l"],
        client=client,
    )
    assert q.affected_variables == ["biomass_g_l"]


def test_classifier_dedups_variables() -> None:
    client = _ScriptedClient({
        "shape": "scoping",
        "affected_runs": [],
        "affected_variables": ["biomass_g_l", "Biomass_G_L", "biomass"],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=["biomass_g_l"],
        client=client,
    )
    assert q.affected_variables == ["biomass_g_l"]


def test_classifier_caps_variables_at_10() -> None:
    """LLM returned 12 valid variable names; cap to 10."""
    big_var_list = [f"var_{i}" for i in range(12)]
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": big_var_list,
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=big_var_list,
        client=client,
    )
    assert len(q.affected_variables) == 10


def test_classifier_caps_runs_at_20() -> None:
    big_run_list = [f"RUN-{i:04d}" for i in range(25)]
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": big_run_list,
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=big_run_list,
        available_variables=[],
        client=client,
    )
    assert len(q.affected_runs) == 20


# ---------- error / fallback paths ----------


def test_classifier_falls_back_when_client_raises() -> None:
    client = _ScriptedClient(RuntimeError("network down"))
    q = classify_user_question(
        text="Why?",
        available_run_ids=["RUN-0001"],
        available_variables=["biomass_g_l"],
        client=client,
    )
    assert q.shape == "open"
    assert q.affected_runs == []
    assert q.affected_variables == []
    assert q.text == "Why?"


def test_classifier_falls_back_in_stub_mode() -> None:
    q = classify_user_question(
        text="anything",
        available_run_ids=["RUN-0001"],
        available_variables=["biomass_g_l"],
        client=None,
    )
    assert q.shape == "open"
    assert q.affected_runs == []


def test_classifier_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        classify_user_question(
            text="",
            available_run_ids=[],
            available_variables=[],
            client=None,
        )


def test_classifier_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValueError):
        classify_user_question(
            text="   ",
            available_run_ids=[],
            available_variables=[],
            client=None,
        )


# ---------- malformed payload tolerance ----------


def test_classifier_handles_non_list_runs_field() -> None:
    """LLM returns a string instead of a list for affected_runs."""
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": "RUN-0001",
        "affected_variables": [],
    })
    q = classify_user_question(
        text="?",
        available_run_ids=["RUN-0001"],
        available_variables=[],
        client=client,
    )
    # Non-list = treat as empty; doesn't crash.
    assert q.affected_runs == []


def test_classifier_handles_non_list_variables_field() -> None:
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": "biomass_g_l",
    })
    q = classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=["biomass_g_l"],
        client=client,
    )
    assert q.affected_variables == []


# ---------- prompt content sanity ----------


def test_classifier_prompt_includes_run_ids_and_variables() -> None:
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": [],
    })
    classify_user_question(
        text="?",
        available_run_ids=["RUN-0001", "RUN-0002"],
        available_variables=["biomass_g_l"],
        client=client,
    )
    [(_system, user_prompt)] = client.calls
    assert "RUN-0001" in user_prompt
    assert "RUN-0002" in user_prompt
    assert "biomass_g_l" in user_prompt
    assert "?" in user_prompt  # the question text itself


def test_classifier_prompt_handles_empty_bundle_metadata() -> None:
    """Stub-fixture bundles often have no run_ids/variables yet."""
    client = _ScriptedClient({
        "shape": "open",
        "affected_runs": [],
        "affected_variables": [],
    })
    classify_user_question(
        text="?",
        available_run_ids=[],
        available_variables=[],
        client=client,
    )
    [(_system, user_prompt)] = client.calls
    assert "(none)" in user_prompt


# ---------- production client constructor ----------


def test_gemini_classifier_client_constructs_without_api_key(monkeypatch) -> None:
    """We don't make a live call — just verify construction doesn't fail
    when GEMINI_API_KEY is missing. Calls would fail later, which is
    fine — classify_user_question catches and falls back."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiQuestionClassifierClient()
    assert client._model
