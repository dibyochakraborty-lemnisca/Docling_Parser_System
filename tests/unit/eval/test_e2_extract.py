from __future__ import annotations

from types import SimpleNamespace

from fermdocs_eval.suites.e2 import extract_fired_axes


def _critique(reasons: list[str]) -> SimpleNamespace:
    return SimpleNamespace(type="critique_filed", reasons=reasons)


def _other(kind: str) -> SimpleNamespace:
    return SimpleNamespace(type=kind, reasons=["[trajectory-axis] should be ignored"])


def test_extract_dedup_and_order() -> None:
    events = [
        _critique(["[trajectory-axis] OD claim outruns evidence"]),
        _critique(["[memory-axis] ignored prior lesson L-xyz"]),
        # duplicate trajectory-axis on a later critique — should not appear twice
        _critique(["[trajectory-axis] same axis again", "[robustness-axis] N=1"]),
    ]
    fired = extract_fired_axes(events)
    assert fired == ["trajectory-axis", "memory-axis", "robustness-axis"]


def test_extract_ignores_non_critique_events() -> None:
    events = [
        _other("hypothesis_synthesized"),  # axis in reasons of non-critique event
        _critique(["[question-axis] ignored the user question"]),
    ]
    fired = extract_fired_axes(events)
    assert fired == ["question-axis"]


def test_extract_handles_dict_events() -> None:
    events = [
        {"type": "critique_filed", "reasons": ["[metadata-axis] missed anomaly F-0012"]},
        {"type": "judge_ruling", "rationale": "[trajectory-axis] not in critique"},
    ]
    fired = extract_fired_axes(events)
    assert fired == ["metadata-axis"]


def test_extract_empty_and_green_critiques() -> None:
    events = [
        _critique([]),  # green-flag critique, no reasons
        _critique(["no axis tag in this reason"]),
    ]
    fired = extract_fired_axes(events)
    assert fired == []
