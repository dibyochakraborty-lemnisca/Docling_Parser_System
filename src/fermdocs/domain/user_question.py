"""User-supplied directive that biases (and later, drives) the pipeline.

Plan ref: plans/2026-05-04-user-question-and-hitl.md (PR-A on caisc-hitl).

Lives in fermdocs.domain so that both characterize, diagnose, and
hypothesis packages can carry it without violating the
characterize → diagnose → hypothesis dependency direction.

A `UserQuestion` is the typed shape we thread through every pipeline
stage when a human arrives with a specific question. The text is the
only required field; `shape`, `affected_variables`, and `affected_runs`
are populated by the LLM-backed classifier in
`fermdocs_hypothesis.question_classifier` and validated against the
bundle's actual run_ids/variables.

raised_by tracks WHO raised the question:
  - "user"             — typed at run start (PR-A)
  - "user_followup"    — typed after a run completes (PR-A2, drive)
  - "operator_mid_debate" — injected mid-debate (PR-B, HITL)

shape drives prompt routing today (PR-A) and topic-set branching later
(PR-A2):
  - scoping     — narrows to a specific run / time / variable
  - mechanistic — proposes a mechanism for the system to test
  - comparative — points at a comparison axis to surface
  - open        — general directive; bottom-up flow with bias
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionShape = Literal["scoping", "mechanistic", "comparative", "open"]
QuestionRaisedBy = Literal["user", "user_followup", "operator_mid_debate"]


class UserQuestion(BaseModel):
    """A directive from the human user that biases the debate.

    Frozen — once a run starts, the question doesn't mutate. Follow-up
    questions create a NEW UserQuestion attached to a derived run.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question, as typed. Not interpreted here.",
    )
    shape: QuestionShape | None = Field(
        default=None,
        description=(
            "Auto-classified by classify_user_question(). None means the"
            " classifier wasn't run yet OR fell back on error."
        ),
    )
    affected_variables: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Variable names extracted from the question text and validated"
            " against the bundle's actual columns. Used by"
            " question_relevance() and the prompt scoping rules."
        ),
    )
    affected_runs: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "RUN-XXXX identifiers extracted from the question text and"
            " validated against the bundle's actual run_ids."
        ),
    )
    raised_by: QuestionRaisedBy = "user"


# -----------------------------------------------------------------------------
# Public helpers shared across pipeline stages
# -----------------------------------------------------------------------------


def question_relevance(
    *,
    affected_variables: list[str] | None = None,
    affected_runs: list[str] | None = None,
    text: str = "",
    question: UserQuestion | None,
) -> float:
    """0.0-1.0 relevance score for a topic / claim / finding shape.

    Caller passes the candidate's own affected_variables/runs/text.
    When `question` is None, returns 0.0 — fully back-compat with
    no-question runs.

    Score components (capped at 1.0):
      - 0.4 if any affected_variable overlaps the question's
      - 0.4 if any affected_run overlaps the question's
      - 0.2 if any candidate variable name appears in question.text
        as a case-insensitive substring (catches free-form references
        the classifier didn't enumerate)

    The `text` arg is reserved for a future free-text-vs-question
    similarity term; today it's accepted but unused so callers don't
    have to change when we add semantic matching.
    """
    if question is None:
        return 0.0
    del text  # reserved; see docstring

    cand_vars = {v.lower() for v in (affected_variables or [])}
    cand_runs = {r.upper() for r in (affected_runs or [])}
    q_vars = {v.lower() for v in question.affected_variables}
    q_runs = {r.upper() for r in question.affected_runs}
    q_text_lower = question.text.lower()

    score = 0.0
    if cand_vars & q_vars:
        score += 0.4
    if cand_runs & q_runs:
        score += 0.4
    if cand_vars and any(v in q_text_lower for v in cand_vars):
        score += 0.2
    return min(score, 1.0)
