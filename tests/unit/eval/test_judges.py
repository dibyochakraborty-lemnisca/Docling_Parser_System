from __future__ import annotations

from fermdocs_eval.judges import _build_head_to_head_prompt, judge_head_to_head


def test_prompt_substitution_no_keyerror() -> None:
    """The JSON spec in the prompt body contains { and } which would crash
    .format(). Builder uses manual replace so this stays robust."""
    out = _build_head_to_head_prompt(
        question="What happened?",
        a_text="Answer A — has braces {nested} and percent %d",
        b_text="Answer B with $variables and \\backslashes",
    )
    assert "What happened?" in out
    assert "Answer A" in out
    assert "Answer B" in out
    # Sanity: original template markers got swapped
    assert "__QUESTION__" not in out
    assert "__A_TEXT__" not in out


def test_judge_empty_inputs_return_error() -> None:
    out = judge_head_to_head(question="", a_text="x", b_text="y")
    assert out["status"] == "error"
    out = judge_head_to_head(question="q", a_text="", b_text="y")
    assert out["status"] == "error"
    out = judge_head_to_head(question="q", a_text="x", b_text="")
    assert out["status"] == "error"
