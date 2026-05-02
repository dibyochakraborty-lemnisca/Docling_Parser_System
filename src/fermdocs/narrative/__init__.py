"""Prose-insight extraction from document narrative blocks.

Plan B Stage 2 (plans/2026-05-03-narrative-insight-extraction.md). Takes
the NarrativeBlocks already produced by DoclingPdfParser and turns them
into typed NarrativeObservation instances via one Gemini call per
document. The result flows through CharacterizationOutput.narrative_observations
to the diagnosis agent (wired in Stage 3).

Design choices:
  - Pure module: NarrativeBlock[] in → NarrativeObservation[] out.
    No file I/O, no PDF parsing — the parser owns that already.
  - One LLM call per document. Caller chunks if the document is huge.
  - Errors swallowed → empty list. Extraction is additive; never blocks
    the rest of the pipeline.
  - Closed NarrativeTag enum (closure_event, deviation, intervention,
    observation, conclusion, protocol_note) constrains the model via
    structured output.
"""

from fermdocs.narrative.extractor import (
    DEFAULT_DOC_CHAR_CAP,
    NarrativeExtractor,
    NarrativeLLMClient,
    extract_narrative_observations,
)

__all__ = [
    "DEFAULT_DOC_CHAR_CAP",
    "NarrativeExtractor",
    "NarrativeLLMClient",
    "extract_narrative_observations",
]
