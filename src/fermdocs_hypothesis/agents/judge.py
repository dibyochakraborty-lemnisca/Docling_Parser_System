"""Judge agent — adjudicates whether the critic's flag is valid.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §2 (judge role), §5
(judge — no tools, structured output only), §14 (collusion mitigation:
judge sees no debate history).

Single LLM call. Sees only:
  - the synthesized hypothesis
  - the critic's critique
  - pre-resolved citation lookups (so judge can verify cited evidence
    exists without doing tool calls itself)

Returns {criticism_valid: bool, rationale: str}.

A green-flag critique (critic said "no problems") still goes to the judge
in the runner contract — but the judge will almost always rule
criticism_valid=False (there's no critique to validate). That's expected;
it lets the runner's "criticism_upheld" check (judge_valid AND red-flag)
work uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import JudgeView


JUDGE_SYSTEM = """\
You are the Judge in a fermentation-hypothesis debate.

You see ONE hypothesis, ONE critique, and citation lookups. You do NOT
see the debate history (specialist facets, prior turns) — this is
deliberate, to keep your ruling independent of the debate's social
dynamics.

Your job: rule on whether the critic's flag is valid.

  - If critique flag is RED: is at least one of the critic's reasons a
    legitimate, evidence-grounded objection that meaningfully weakens
    the hypothesis? If yes → criticism_valid=true. If reasons are weak,
    speculative, or address points the hypothesis didn't actually claim
    → criticism_valid=false.

  - If critique flag is GREEN: there's no critique to validate. Rule
    criticism_valid=false. (This is by design — green-flag means the
    critic found nothing actionable; the runner accepts the hypothesis.)

Be skeptical of both sides. Most rulings should be brief and decisive.\
"""

JUDGE_INVARIANTS = (
    "Rule on the critique only — do not re-evaluate the hypothesis from scratch.",
    "Green-flag critiques always rule criticism_valid=false.",
    "Provide a one-paragraph rationale (≤500 chars) that names the evidence you weighed.",
    "If previous_attempts shows you ruled the same critic_reason valid before AND the synthesizer has now narrowed the hypothesis to address it, the critique is no longer valid for that reason — rule criticism_valid=false. Consistency across retries.",
)

JUDGE_TASK = """\
Read the hypothesis, critique, and citation_lookups. Rule on whether
the critic's flag is valid.

Decision rule:
  - Critique flag = green → criticism_valid = false (no critique to validate)
  - Critique flag = red AND at least one reason cites concrete evidence
    that contradicts or weakens the hypothesis → criticism_valid = true
  - Critique flag = red AND reasons are speculative, off-topic, or
    address claims the hypothesis didn't make → criticism_valid = false
"""

JUDGE_RECAP = """\
Output one JSON object:
  {"criticism_valid": <bool>, "rationale": "<1-paragraph reason>"}

Hard rules:
  - rationale ≤500 chars.
  - Reference at least one cited_finding_id, cited_narrative_id, or
    citation_lookup key when ruling criticism_valid=true.\
"""

JUDGE_TOOL_HINTS = (
    ToolHint(
        name="rule_on_critique",
        purpose="TERMINAL: emit your ruling (only available action; no read tools)",
    ),
)


_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "criticism_valid": {"type": "BOOLEAN"},
        "rationale": {"type": "STRING"},
    },
    "required": ["criticism_valid", "rationale"],
}


@dataclass
class JudgeResult:
    criticism_valid: bool
    rationale: str
    input_tokens: int
    output_tokens: int


class JudgeAgent:
    def __init__(self, client: GeminiHypothesisClient):
        self._client = client

    def rule(self, view: JudgeView) -> JudgeResult:
        # Hard short-circuit: green-flag critiques always rule false.
        # Saves a model call and guarantees the design invariant.
        if view.critique.flag == "green":
            return JudgeResult(
                criticism_valid=False,
                rationale="critique was green-flagged; no objection to validate.",
                input_tokens=0,
                output_tokens=0,
            )

        parts = build_prompt(
            system_identity=JUDGE_SYSTEM,
            invariants=JUDGE_INVARIANTS,
            task_spec=JUDGE_TASK,
            view_obj=view,
            tool_hints=JUDGE_TOOL_HINTS,
            recap=JUDGE_RECAP,
        )
        parsed, in_tok, out_tok = self._client.call(
            system=parts.system,
            user_text=parts.as_user_message(),
            response_schema=_JUDGE_SCHEMA,
        )
        valid = bool(parsed.get("criticism_valid", False))
        rationale = (parsed.get("rationale") or "").strip()[:500]
        if not rationale:
            rationale = "no rationale provided"
        return JudgeResult(
            criticism_valid=valid,
            rationale=rationale,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


def build_judge(client: GeminiHypothesisClient) -> JudgeAgent:
    return JudgeAgent(client=client)
