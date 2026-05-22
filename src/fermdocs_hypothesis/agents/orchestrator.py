"""Orchestrator agent — picks a topic from the ranker's top-K.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §2 (orchestrator role),
§5 (orchestrator tool surface).

Design (deliberately small):

  - The deterministic ranker has already computed top-K topics. Orchestrator
    LLM only chooses among them and provides a rationale.
  - The LLM CAN also add an open question if it wants the debate to surface
    something not in the seed topics (rare in v0).
  - exit_stage is available too — if every top topic is already in
    used_topic_ids OR open questions list is empty, the LLM may bail.

Why not let the LLM pick freely from anywhere? The plan's §8 mandates a
deterministic ranker so loops can't degenerate. The LLM gets a small,
ranked menu — it can't invent topics.

Returned action shape:
  {
    "action": "select_topic" | "add_open_question" | "exit_stage",
    "topic_id": "T-NNNN" (when select_topic),
    "rationale": str (when select_topic),
    "question": str (when add_open_question),
    "tags": [str] (when add_open_question),
    "exit_reason": str (when exit_stage),
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fermdocs_hypothesis.llm_clients import _ORCHESTRATOR_SCHEMA, GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import OrchestratorView

ORCHESTRATOR_SYSTEM = """\
You are the Orchestrator of a fermentation-hypothesis debate.

Your job each turn: pick the next debate topic from the ranker's top-K,
or — if there are no good topics left — exit the stage.

You operate AFTER characterization and diagnosis stages. Both have
emitted frozen artifacts. Your only job is to sequence what the
specialists debate. You do not produce hypotheses yourself; specialists
do that.

Pick the highest-scored unattempted topic unless its rationale is poor.
If two topics are very close in score, prefer the one whose summary
gives specialists the most leverage (richer citations, clearer variable
scope).

You may ALSO add an open question if you see a critical data gap that no
seed topic captures. Use this sparingly — open questions become synthetic
topics in subsequent turns.

Stay observational. Do not opine on causality. Do not reference
information not in the view.\
"""

ORCHESTRATOR_INVARIANTS = (
    "You may only select topic_ids that appear in `top_topics` of the view.",
    "Provide a one-sentence rationale for each selection.",
    "If no topics remain or you have ≥2 accepted hypotheses already, exit_stage.",
    "Never invent a topic_id.",
)

ORCHESTRATOR_TASK = """\
Pick the next debate topic. Output exactly one action.

Decision policy:
  1. If accepted_hypotheses_so_far has ≥2 entries → exit_stage("consensus_reached").
  2. Else if top_topics is empty → exit_stage("no_topics_left").
  3. Else select_topic with the topic_id of the strongest unattempted candidate
     in top_topics, plus a one-sentence rationale.
  4. Optionally (rarely): add_open_question if a critical data gap is missing.\
"""

ORCHESTRATOR_RECAP = """\
Output one JSON object matching the orchestrator schema.

Hard rules:
  - topic_id MUST be one of the IDs in view.top_topics.
  - For select_topic: rationale is required, ≤200 chars.
  - For add_open_question: tags must be lowercased domain hints (e.g. ["DO","kLa"]).
  - For exit_stage: exit_reason ∈ {"consensus_reached","no_topics_left",
    "budget_exhausted","max_turns_reached"}.\
"""

ORCHESTRATOR_TOOL_HINTS = (
    ToolHint(
        name="select_topic",
        purpose="pick a topic_id from view.top_topics for this turn",
    ),
    ToolHint(
        name="add_open_question",
        purpose="register a question for later debate (tags drive routing)",
    ),
    ToolHint(
        name="exit_stage",
        purpose="end the stage with a reason",
    ),
)


@dataclass
class OrchestratorAction:
    action: str
    topic_id: str | None = None
    rationale: str | None = None
    question: str | None = None
    tags: list[str] | None = None
    exit_reason: str | None = None


class OrchestratorAgent:
    """One-call wrapper. The runner invokes `decide(view) -> (action, in_tok, out_tok)`."""

    def __init__(self, client: GeminiHypothesisClient):
        self._client = client

    def decide(self, view: OrchestratorView) -> tuple[OrchestratorAction, int, int]:
        parts = build_prompt(
            system_identity=ORCHESTRATOR_SYSTEM,
            invariants=ORCHESTRATOR_INVARIANTS,
            task_spec=ORCHESTRATOR_TASK,
            view_obj=view,
            tool_hints=ORCHESTRATOR_TOOL_HINTS,
            recap=ORCHESTRATOR_RECAP,
        )
        parsed, in_tok, out_tok = self._client.call(
            system=parts.system,
            user_text=parts.as_user_message(),
            response_schema=_ORCHESTRATOR_SCHEMA,
        )
        action = _parse_action(parsed, view)
        return action, in_tok, out_tok


def _parse_action(parsed: dict[str, Any], view: OrchestratorView) -> OrchestratorAction:
    action = parsed.get("action", "").strip()
    if action == "select_topic":
        tid = (parsed.get("topic_id") or "").strip()
        # Hard guard: topic_id must be in view.top_topics. If LLM hallucinated,
        # auto-correct to the highest-scored topic.
        valid_ids = {t.topic_id for t in view.top_topics}
        if tid not in valid_ids:
            if view.top_topics:
                tid = view.top_topics[0].topic_id
            else:
                return OrchestratorAction(
                    action="exit_stage",
                    exit_reason="no_topics_left",
                )
        return OrchestratorAction(
            action="select_topic",
            topic_id=tid,
            rationale=(parsed.get("rationale") or "")[:200] or "ranker top pick",
        )
    if action == "add_open_question":
        return OrchestratorAction(
            action="add_open_question",
            question=parsed.get("question") or "",
            tags=list(parsed.get("tags") or []),
        )
    if action == "exit_stage":
        return OrchestratorAction(
            action="exit_stage",
            exit_reason=parsed.get("exit_reason") or "no_topics_left",
        )
    # Unknown action → fall back to safest exit
    return OrchestratorAction(action="exit_stage", exit_reason="no_topics_left")
