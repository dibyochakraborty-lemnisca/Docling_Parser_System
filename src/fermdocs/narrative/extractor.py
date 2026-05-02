"""Gemini-backed prose-insight extractor.

Takes NarrativeBlocks (from DoclingPdfParser or any other source) and emits
NarrativeObservation instances tagged with a closed taxonomy. One Gemini
call per document.

The extractor is a pure module — it does not touch the file system or call
the parser. The ingestion pipeline owns that handoff and feeds blocks in.

Error posture (from plan §3): extraction is ADDITIVE. A document with no
extractable insights or a failing extractor produces an empty list and
downstream behavior is identical to today. Never raises across the
ingestion boundary.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol
from uuid import UUID

from fermdocs.domain.models import NarrativeBlock
from fermdocs_characterize.schema import (
    NarrativeObservation,
    NarrativeSourceLocator,
    NarrativeTag,
)

_log = logging.getLogger(__name__)

DEFAULT_DOC_CHAR_CAP = 200_000
"""Hard cap on document text passed to the LLM in one call. Larger docs
are truncated and a warning is logged. Real fermentation reports are
typically 5-50 KB; carotenoid case is ~30 KB. Cap protects against
runaway prompts and matches plan §3."""

DEFAULT_LLM_TIMEOUT_S = 60
"""Wall-clock timeout for the extractor's single Gemini call."""

DEFAULT_GEMINI_MODEL = "gemini-3-pro"


class NarrativeLLMClient(Protocol):
    """Minimal protocol so tests can supply a scripted client.

    `call(rendered_blocks: str) -> list[dict]` — returns a list of dicts
    matching the NarrativeObservation shape (without `narrative_id` —
    extractor assigns those after).
    """

    def call(self, rendered_blocks: str) -> list[dict[str, Any]]: ...


# -----------------------------------------------------------------------------
# Rendering: NarrativeBlock[] → text the LLM sees
# -----------------------------------------------------------------------------


def _render_blocks(blocks: list[NarrativeBlock], char_cap: int) -> tuple[str, bool]:
    """Render blocks to a single string, indexed for citation.

    Returns (text, truncated_flag). The LLM sees one line per block:

        [BLOCK 0 | page=4 | section=Results | type=paragraph]
        terminated at 82h, white cells observed during centrifugation
        ---

    Block index is the stable ID the LLM uses for source_locator.paragraph_index.
    """
    parts: list[str] = []
    used = 0
    truncated = False
    for idx, block in enumerate(blocks):
        loc = block.locator or {}
        header = (
            f"[BLOCK {idx} | page={loc.get('page')} | "
            f"section={loc.get('section')} | type={block.type.value}]"
        )
        body = (block.text or "").strip()
        if not body:
            continue
        chunk = f"{header}\n{body}\n---\n"
        if used + len(chunk) > char_cap:
            truncated = True
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts), truncated


# -----------------------------------------------------------------------------
# Extractor
# -----------------------------------------------------------------------------


class NarrativeExtractor:
    """Synchronous Gemini-driven prose extractor.

    Construct with an explicit `client` for tests; production callers use the
    default `GeminiNarrativeClient` via the no-arg constructor.
    """

    def __init__(
        self,
        client: NarrativeLLMClient | None = None,
        *,
        model: str | None = None,
        char_cap: int = DEFAULT_DOC_CHAR_CAP,
    ) -> None:
        self._client = client
        self._model = (
            model
            or os.environ.get("FERMDOCS_NARRATIVE_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        )
        self._char_cap = char_cap

    def extract(
        self,
        blocks: list[NarrativeBlock],
        *,
        characterization_id: UUID,
    ) -> list[NarrativeObservation]:
        """Run the extractor. Never raises across the public boundary.

        Args:
            blocks: NarrativeBlocks from the parser (paragraphs, headings, etc).
                Empty list returns empty list — no LLM call.
            characterization_id: Used to namespace the emitted narrative_ids
                as '<char_id>:N-NNNN', matching the existing finding pattern.

        Returns:
            list[NarrativeObservation]. Empty on extractor failure or empty input.
        """
        if not blocks:
            return []
        rendered, truncated = _render_blocks(blocks, self._char_cap)
        if not rendered.strip():
            return []
        if truncated:
            _log.warning(
                "narrative extractor: input >%d chars, truncated. Larger documents"
                " should be split before extraction.",
                self._char_cap,
            )

        client = self._client or self._build_default_client()
        try:
            raw_items = client.call(rendered)
        except Exception as exc:
            _log.warning(
                "narrative extractor: client call failed (%s: %s) — emitting empty list",
                exc.__class__.__name__,
                str(exc)[:200],
            )
            return []

        if not isinstance(raw_items, list):
            _log.warning(
                "narrative extractor: client returned non-list %s — emitting empty",
                type(raw_items).__name__,
            )
            return []

        return self._materialize(raw_items, characterization_id=characterization_id)

    def _build_default_client(self) -> NarrativeLLMClient:
        return GeminiNarrativeClient(model=self._model)

    def _materialize(
        self,
        raw_items: list[Any],
        *,
        characterization_id: UUID,
    ) -> list[NarrativeObservation]:
        out: list[NarrativeObservation] = []
        seen_texts: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            obs = self._coerce_one(
                raw,
                seq=len(out) + 1,
                characterization_id=characterization_id,
                seen_texts=seen_texts,
            )
            if obs is not None:
                out.append(obs)
                seen_texts.add(obs.text.strip().lower())
        return out

    def _coerce_one(
        self,
        raw: dict[str, Any],
        *,
        seq: int,
        characterization_id: UUID,
        seen_texts: set[str],
    ) -> NarrativeObservation | None:
        text = (raw.get("text") or "").strip()
        if not text:
            return None
        if text.lower() in seen_texts:
            return None  # de-dup verbatim repeats

        tag_raw = raw.get("tag")
        try:
            tag = NarrativeTag(tag_raw) if tag_raw is not None else None
        except ValueError:
            _log.info("narrative extractor: dropping unknown tag %r", tag_raw)
            return None
        if tag is None:
            return None

        loc_raw = raw.get("source_locator") or {}
        locator = NarrativeSourceLocator(
            page=_coerce_int(loc_raw.get("page")),
            section=_coerce_str_or_none(loc_raw.get("section")),
            paragraph_index=_coerce_int(loc_raw.get("paragraph_index")),
            char_offset=_coerce_int(loc_raw.get("char_offset")),
        )

        confidence = _coerce_confidence(raw.get("confidence"))
        try:
            return NarrativeObservation(
                narrative_id=f"{characterization_id}:N-{seq:04d}",
                tag=tag,
                text=text,
                source_locator=locator,
                run_id=_coerce_str_or_none(raw.get("run_id")),
                time_h=_coerce_float(raw.get("time_h")),
                affected_variables=_coerce_str_list(raw.get("affected_variables")),
                confidence=confidence,
                extraction_model=self._model,
            )
        except Exception as exc:
            _log.info(
                "narrative extractor: dropping malformed item (%s: %s)",
                exc.__class__.__name__,
                str(exc)[:200],
            )
            return None


# -----------------------------------------------------------------------------
# Coercion helpers — defensive, never raise
# -----------------------------------------------------------------------------


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _coerce_confidence(value: Any) -> float:
    """Capped at 0.85 (LLM-judged posture). Defaults to 0.6 if unparseable."""
    f = _coerce_float(value)
    if f is None:
        return 0.6
    return max(0.0, min(0.85, f))


# -----------------------------------------------------------------------------
# Convenience top-level helper
# -----------------------------------------------------------------------------


def extract_narrative_observations(
    blocks: list[NarrativeBlock],
    *,
    characterization_id: UUID,
    client: NarrativeLLMClient | None = None,
    model: str | None = None,
) -> list[NarrativeObservation]:
    """One-shot helper. Equivalent to NarrativeExtractor(...).extract(...)."""
    return NarrativeExtractor(client=client, model=model).extract(
        blocks, characterization_id=characterization_id
    )


# -----------------------------------------------------------------------------
# Default Gemini client
# -----------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a fermentation report reader. Your job is to extract OBSERVATIONAL\n"
    "insights from the prose blocks of a batch / fermentation report into a\n"
    "structured list. Tables and numerical data are handled by a separate\n"
    "deterministic pipeline; you do NOT process numbers — you process the\n"
    "operator's, scientist's, and report-author's STATEMENTS.\n\n"
    "TAG TAXONOMY (closed; pick exactly one per observation):\n"
    "  - closure_event: the run ended for a notable reason ('terminated at\n"
    "    82h due to onset of cell death', 'cultivation completed at 96h')\n"
    "  - deviation: an observed departure from plan ('DO dropped to 20% in\n"
    "    late phase', 'pH excursion at 48h')\n"
    "  - intervention: a deliberate action taken during the run ('200 mL IPM\n"
    "    added at 24h', 'antifoam added at 48h')\n"
    "  - observation: a non-deviation factual statement about what was seen\n"
    "    ('white cells visible during centrifugation', 'foaming reported at 36h')\n"
    "  - conclusion: a stated outcome or judgement ('yield 30% below target',\n"
    "    'culture stability maintained until end')\n"
    "  - protocol_note: method-section detail (use sparingly; only when an\n"
    "    insight is genuinely procedural and may matter for diagnosis)\n\n"
    "RULES:\n"
    "  1. Quote verbatim. The `text` field is the exact source phrase, lightly\n"
    "     normalized (whitespace collapse only). Never paraphrase.\n"
    "  2. One tag per observation. If a sentence carries multiple tags,\n"
    "     produce multiple observations.\n"
    "  3. Source locator: copy the BLOCK index from the input header into\n"
    "     source_locator.paragraph_index. If a `page=` value is in the header,\n"
    "     copy it. If `section=` is present, copy it.\n"
    "  4. Confidence ≤ 0.85. The pipeline caps higher values automatically.\n"
    "     Use 0.8 for clearly-stated insights; lower for hedged language.\n"
    "  5. run_id and time_h: extract whenever the prose attributes the\n"
    "     observation to a specific run/time. SECTION HEADERS COUNT as\n"
    "     attribution: a heading like 'BATCH-01 REPORT', 'Batch 04', or\n"
    "     'B05' establishes a run scope, and every subsequent observation\n"
    "     in that section carries that run_id (use the canonical form, e.g.\n"
    "     'BATCH-01', 'BATCH-04', 'BATCH-05'). When you encounter a new\n"
    "     batch header, switch run_id for everything that follows until\n"
    "     another batch header appears. Use null only when the entire\n"
    "     document has no batch grouping or the observation predates the\n"
    "     first batch header.\n"
    "  6. affected_variables: list canonical variable names when the prose\n"
    "     plainly names them (biomass, glucose, ethanol, DO, pH, viability,\n"
    "     etc.). Empty list when no clear variable is named.\n"
    "  7. Causal hypotheses are NOT observations. Skip 'X happened BECAUSE of Y'.\n"
    "  8. Skip the same statement twice. The pipeline dedups verbatim repeats\n"
    "     anyway, but help out by being terse.\n"
    "  9. Empty list is acceptable — only emit observations that genuinely\n"
    "     carry information. A minimum of 0 observations is fine.\n\n"
    "OUTPUT: a JSON list of objects with fields {tag, text, source_locator,\n"
    "run_id, time_h, affected_variables, confidence}. Nothing else."
)

_GEMINI_NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "tag": {
                "type": "STRING",
                "enum": [t.value for t in NarrativeTag],
            },
            "text": {"type": "STRING"},
            "source_locator": {
                "type": "OBJECT",
                "properties": {
                    "page": {"type": "INTEGER", "nullable": True},
                    "section": {"type": "STRING", "nullable": True},
                    "paragraph_index": {"type": "INTEGER", "nullable": True},
                    "char_offset": {"type": "INTEGER", "nullable": True},
                },
                "nullable": True,
            },
            "run_id": {"type": "STRING", "nullable": True},
            "time_h": {"type": "NUMBER", "nullable": True},
            "affected_variables": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "nullable": True,
            },
            "confidence": {"type": "NUMBER", "nullable": True},
        },
        "required": ["tag", "text"],
    },
}


class GeminiNarrativeClient:
    """Default NarrativeLLMClient backed by Google Gemini structured output."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_NARRATIVE_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, rendered_blocks: str) -> list[dict[str, Any]]:
        from google import genai
        from google.genai import types

        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not set; narrative extractor requires it")
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Extract narrative observations from the following"
                                " prose blocks of a fermentation report. Follow the"
                                " system instructions exactly.\n\n"
                                "BLOCKS:\n" + rendered_blocks
                            )
                        }
                    ],
                },
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_GEMINI_NARRATIVE_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            return []
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []
        return parsed
