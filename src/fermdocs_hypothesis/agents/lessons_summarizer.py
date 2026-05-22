"""LessonsSummarizerAgent — compresses recurring critic complaints into a
short digest that retrying agents see in their views.

Plan ref: feedback-loop §4 (this PR). Sits outside the orchestrator →
specialist → synthesizer → critic → judge cycle. Invoked by the runner on
retry phases when enough new critic reasons have accumulated since the
last digest (cache key: source_reason_count).

Single-shot LLM call, no tool loop. Output is a short prose digest the
synthesizer/critic/judge prompts can paste verbatim into their context.

Stub mode: when client is None, returns a deterministic digest
("DETERMINISTIC[k]: r1 | r2 | ...") so tests stay reproducible without
mocking Gemini. The same code path runs in prod with a real client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import Lesson, LessonsDigest


class LessonsView(BaseModel):
    """View handed to LessonsSummarizerAgent.

    `recent_critic_reasons` is the last-K (K=20 by default in
    state.all_critic_reasons) verbatim critic reasons across all topics.
    Order: oldest-first within the window so the LLM can spot drift.
    """

    model_config = ConfigDict(frozen=True)

    recent_critic_reasons: list[str] = Field(default_factory=list)


LESSONS_SYSTEM = """\
You are the Lessons Summarizer in a fermentation-hypothesis debate.

You are given a list of recent critic complaints from a multi-topic
debate. Your job: identify the 3-5 RECURRING patterns the synthesizer
keeps falling into, and write each as a short rule the synthesizer should
follow on its next attempt.

Examples of useful patterns to surface:
  - "Repeatedly inferring causal absence from documented absence (3
    occurrences across topics): documentary silence is not proof."
  - "Repeatedly extending a single-batch citation to multi-batch claims
    (2 occurrences): scope claims to the cited batch only."

Avoid:
  - One-off complaints (only summarize patterns that appear ≥2 times).
  - Generic advice ("be more careful with citations") — be specific.
  - Repeating the critic's wording verbatim — distill the pattern.

Output 3-5 lessons. If fewer than 3 recurring patterns exist, return
fewer. Empty list is acceptable when nothing recurs.\
"""

LESSONS_INVARIANTS = (
    "Only surface patterns that appear ≥2 times across the input.",
    "Each lesson is one sentence, ≤200 chars, actionable.",
    "Distill the pattern; do not quote the critic verbatim.",
)

LESSONS_TASK = """\
Read the recent_critic_reasons list. Identify recurring patterns. Emit
{lessons: [str]} with 3-5 entries (or fewer if patterns are scarce).
"""

LESSONS_RECAP = """\
Output one JSON object: {"lessons": ["...", "..."]}.

Hard rules:
  - 0-5 lessons.
  - Each lesson ≤200 chars.
  - No lesson may quote a single critic reason verbatim — distill.\
"""

LESSONS_TOOL_HINTS: tuple[ToolHint, ...] = ()


_LESSONS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "lessons": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["lessons"],
}


@dataclass
class LessonsResult:
    digest: LessonsDigest
    input_tokens: int
    output_tokens: int


class LessonsSummarizerAgent:
    """LLM-backed summarizer. Pass client=None for deterministic stub mode.

    Stub mode is for tests that need the runner to call the summarizer
    without exercising Gemini — the digest content is irrelevant to the
    test, only the call/cache behavior is. A separate test exercises the
    real prompt with a mocked Gemini response.
    """

    def __init__(
        self,
        client: GeminiHypothesisClient | None,
        *,
        run_id: str | None = None,
    ):
        self._client = client
        # run_id is used to mint stable lesson_ids: L-<run_short>-<NNNN>.
        # When None (legacy callers), we fall back to a synthetic prefix.
        self._run_id = run_id
        self._lesson_counter = 0

    def _mint_lesson_id(self) -> str:
        """Stable lesson_id for memory-layer round-trip (D2)."""
        self._lesson_counter += 1
        run_short = (self._run_id or "UNKNOWN").split("-")[0][:8] or "RUN"
        return f"L-{run_short}-{self._lesson_counter:04d}"

    def summarize(
        self, view: LessonsView, *, source_reason_count: int
    ) -> LessonsResult:
        if self._client is None:
            return self._stub_summarize(view, source_reason_count)

        parts = build_prompt(
            system_identity=LESSONS_SYSTEM,
            invariants=LESSONS_INVARIANTS,
            task_spec=LESSONS_TASK,
            view_obj=view,
            tool_hints=LESSONS_TOOL_HINTS,
            recap=LESSONS_RECAP,
        )
        parsed, in_tok, out_tok = self._client.call(
            system=parts.system,
            user_text=parts.as_user_message(),
            response_schema=_LESSONS_SCHEMA,
        )
        # Truncate at 200 chars to match the LESSONS_INVARIANTS cap the LLM
        # is told to honor. The Lesson schema accepts up to 400 as a safety
        # margin but the prompt-side contract is 200.
        raw_lessons = [str(l).strip()[:200] for l in (parsed.get("lessons") or []) if l]
        # Mint a Lesson object per surfaced pattern (D2). Tags are
        # heuristic-derived on first pass — the LLM doesn't emit them
        # yet; we'd need a richer prompt schema for that. Phase 2.
        lessons = [
            Lesson(lesson_id=self._mint_lesson_id(), text=text, tags=[])
            for text in raw_lessons
        ]
        digest_text = self._format_digest(raw_lessons)
        digest = LessonsDigest(
            digest=digest_text or "(no recurring patterns surfaced)",
            source_reason_count=source_reason_count,
            computed_at_event_idx=0,
            lessons=lessons,
        )
        return LessonsResult(digest=digest, input_tokens=in_tok, output_tokens=out_tok)

    def _stub_summarize(
        self, view: LessonsView, source_reason_count: int
    ) -> LessonsResult:
        # Deterministic, traceable stub: encodes inputs so tests can assert
        # on exact strings if they want. No tokens billed. Empty input
        # gets an explicit "(empty)" suffix so LessonsDigest.min_length
        # is satisfied AND so the runner can distinguish "we ran with
        # nothing to chew on" from a real digest.
        if not view.recent_critic_reasons:
            text = "DETERMINISTIC[0]: (empty)"
            structured: list[Lesson] = []
        else:
            joined = " | ".join(view.recent_critic_reasons[:5])
            text = f"DETERMINISTIC[{len(view.recent_critic_reasons)}]: {joined}"
            # One Lesson per cited reason, capped at 5 to match the join.
            structured = [
                Lesson(
                    lesson_id=self._mint_lesson_id(),
                    text=str(r)[:400],
                    tags=[],
                )
                for r in view.recent_critic_reasons[:5]
            ]
        digest = LessonsDigest(
            digest=text,
            source_reason_count=source_reason_count,
            computed_at_event_idx=0,
            lessons=structured,
        )
        return LessonsResult(digest=digest, input_tokens=0, output_tokens=0)

    @staticmethod
    def _format_digest(lessons: list[str]) -> str:
        if not lessons:
            return ""
        return "\n".join(f"  - {l}" for l in lessons)


def build_lessons_summarizer(
    client: GeminiHypothesisClient | None,
    *,
    run_id: str | None = None,
) -> LessonsSummarizerAgent:
    return LessonsSummarizerAgent(client=client, run_id=run_id)
