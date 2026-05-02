"""Identity extractor: pick a ProcessIdentity for an experiment from prose.

The extractor reads the dossier's narrative blocks and returns a
ProcessIdentity. The LLM picks a registered process or null; the result
is gated through a layered safety stack:

    narrative blocks empty?
       YES -> UNKNOWN, no LLM call
       NO  v
    LLM picks process_id from the registry (closed whitelist)
       null / off-list / error -> UNKNOWN
       on-list                 v
    every cited evidence span is a verbatim substring of the source
       no  -> UNKNOWN
       yes v
    variable_fingerprint check against the dossier's present variables
       required missing or forbidden present -> UNKNOWN
       ok                                    v
    confidence is capped at LLM_CONFIDENCE_CAP (0.85)

This composition is the "deterministic where possible, LLM where
necessary" pattern: the LLM picks among options; the registry, the
substring check, and the fingerprint decide whether the pick is allowed.

The extractor never raises. All failure paths return a UNKNOWN identity
with a `rationale` describing why.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from fermdocs.domain.models import (
    EvidenceLocator,
    IdentityProvenance,
    NarrativeBlock,
    ProcessIdentity,
    ScaleInfo,
)
from fermdocs.mapping.evidence_gated_llm import (
    LLM_CONFIDENCE_CAP,
    verify_substring_evidence,
)
from fermdocs.mapping.process_registry import (
    ProcessRegistry,
    ProcessRegistryEntry,
    fingerprint_check,
)


_SYSTEM_PROMPT = (
    "You classify the process described in a fermentation experiment.\n\n"
    "You will be given:\n"
    "  - a CLOSED LIST of allowed process IDs with their organism,\n"
    "    product, and known aliases\n"
    "  - prose paragraphs from the experiment's source documents\n\n"
    "Your job is to pick exactly one process_id from the list, or null"
    " if no entry matches. Rules:\n"
    "  - You may ONLY emit a process_id from the provided list. Do not"
    " invent IDs.\n"
    "  - For each non-null field you emit, you MUST cite a verbatim"
    " substring of one of the input paragraphs as evidence.\n"
    "  - Each evidence span is <= 200 characters.\n"
    "  - If the prose is ambiguous, return null. Don't guess.\n"
    "  - Do not infer organism or product from variable names alone."
    " Variables are checked separately by a fingerprint validator.\n"
    "  - confidence is your subjective confidence (0..1). It will be"
    " capped at 0.85 downstream regardless of what you emit."
)


class IdentityLLMClient(Protocol):
    """Minimal protocol so tests can supply a scripted client."""

    def call(self, system: str, user: str) -> dict[str, Any]: ...


class IdentityExtractor:
    """Wraps an LLMClient with the safety stack documented in the module
    docstring. Tests instantiate this with a scripted client; production
    instantiates with AnthropicClient or GeminiClient (TBD when we wire
    a real call).
    """

    def __init__(
        self,
        registry: ProcessRegistry,
        client: IdentityLLMClient | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        self._registry = registry
        self._client = client
        self._timeout_s = timeout_s

    def extract(
        self,
        narrative_blocks: list[NarrativeBlock],
        present_variables: set[str],
        *,
        file_id: str | None = None,
    ) -> ProcessIdentity:
        if not narrative_blocks:
            return ProcessIdentity(
                provenance=IdentityProvenance.UNKNOWN,
                rationale="no narrative blocks; LLM not called",
            )

        if self._client is None:
            return ProcessIdentity(
                provenance=IdentityProvenance.UNKNOWN,
                rationale="no LLM client configured",
            )

        try:
            payload = self._client.call(
                _SYSTEM_PROMPT,
                _render_user_prompt(self._registry, narrative_blocks),
            )
        except Exception as exc:
            return ProcessIdentity(
                provenance=IdentityProvenance.UNKNOWN,
                rationale=f"LLM call failed: {exc.__class__.__name__}",
            )

        return _validate_payload(
            payload=payload,
            registry=self._registry,
            narrative_blocks=narrative_blocks,
            present_variables=present_variables,
            default_file_id=file_id,
        )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _render_user_prompt(
    registry: ProcessRegistry, blocks: list[NarrativeBlock]
) -> str:
    process_lines = []
    for entry in registry.processes:
        organism_aliases = ", ".join(entry.aliases.organism) or "-"
        product_aliases = ", ".join(entry.aliases.product) or "-"
        process_lines.append(
            f"- id: {entry.id}\n"
            f"  organism: {entry.organism}  (aliases: {organism_aliases})\n"
            f"  product: {entry.product}  (aliases: {product_aliases})\n"
            f"  process_family: {entry.process_family}"
        )
    process_block = "\n".join(process_lines)

    paragraphs = [
        {
            "paragraph_idx": b.locator.get("paragraph_idx", -1),
            "page": b.locator.get("page"),
            "text": b.text,
        }
        for b in blocks
    ]
    return (
        f"ALLOWED PROCESSES:\n{process_block}\n\n"
        f"PARAGRAPHS:\n{json.dumps(paragraphs, ensure_ascii=False)}\n\n"
        "Return JSON with keys: process_id (string|null), organism (string|null),"
        " product (string|null), process_family (string|null),"
        " scale_volume_l (number|null), confidence (number 0..1),"
        " rationale (string), evidence (list of objects each with keys"
        " paragraph_idx (integer) and span_text (string))."
    )


def _validate_payload(
    *,
    payload: dict[str, Any],
    registry: ProcessRegistry,
    narrative_blocks: list[NarrativeBlock],
    present_variables: set[str],
    default_file_id: str | None,
) -> ProcessIdentity:
    process_id = payload.get("process_id")
    by_id = registry.by_id()

    if not process_id:
        return ProcessIdentity(
            provenance=IdentityProvenance.UNKNOWN,
            rationale=str(payload.get("rationale") or "LLM returned null process_id"),
        )

    if process_id not in by_id:
        return ProcessIdentity(
            provenance=IdentityProvenance.UNKNOWN,
            rationale=f"off-whitelist process_id {process_id!r}",
        )

    entry = by_id[process_id]

    # Evidence must be substrings of one of the input paragraphs.
    raw_evidence = payload.get("evidence") or []
    by_paragraph = {
        b.locator.get("paragraph_idx", -1): b for b in narrative_blocks
    }
    locators: list[EvidenceLocator] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        para_idx = item.get("paragraph_idx")
        span = item.get("span_text") or ""
        if para_idx not in by_paragraph:
            return ProcessIdentity(
                provenance=IdentityProvenance.UNKNOWN,
                rationale=f"evidence cites unknown paragraph_idx {para_idx}",
            )
        block = by_paragraph[para_idx]
        ok, reason = verify_substring_evidence(span, block.text, value=None)
        if not ok:
            return ProcessIdentity(
                provenance=IdentityProvenance.UNKNOWN,
                rationale=f"evidence rejected: {reason}",
            )
        locators.append(
            EvidenceLocator(
                file_id=block.locator.get("file_id") or default_file_id or "",
                paragraph_idx=int(para_idx),
                span_text=span,
                span_start=block.text.find(span) if span else None,
            )
        )

    # Fingerprint check: required variables must be present, forbidden absent.
    ok, reason = fingerprint_check(entry, present_variables)
    if not ok:
        return ProcessIdentity(
            process_id=None,
            provenance=IdentityProvenance.UNKNOWN,
            rationale=f"fingerprint mismatch for {process_id!r}: {reason}",
            evidence_locators=locators[:5],
        )

    # Confidence cap.
    raw_conf = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(LLM_CONFIDENCE_CAP, confidence))

    scale_volume = payload.get("scale_volume_l")
    scale = (
        ScaleInfo(volume_l=float(scale_volume))
        if isinstance(scale_volume, (int, float))
        else None
    )

    return ProcessIdentity(
        process_id=process_id,
        organism=payload.get("organism") or entry.organism,
        product=payload.get("product") or entry.product,
        process_family=payload.get("process_family") or entry.process_family,
        scale=scale,
        confidence=confidence,
        provenance=IdentityProvenance.LLM_WHITELISTED,
        evidence_locators=locators[:5],
        rationale=payload.get("rationale"),
    )
