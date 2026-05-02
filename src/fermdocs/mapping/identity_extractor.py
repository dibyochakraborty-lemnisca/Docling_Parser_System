"""Identity extractor: pick a ProcessIdentity for an experiment from prose.

The extractor reads the dossier's narrative blocks and returns a
ProcessIdentity carrying TWO layers:

    process.observed:    organism, product, scale, family hint
                         Populated whenever the LLM finds them in prose.
                         Substring-evidence verified per field.
                         Survives even when the registry has no match.

    process.registered:  registry process_id, plus its provenance.
                         Populated only on registry hit + fingerprint pass.
                         Stays UNKNOWN for processes outside the registry.

Failure of the registered layer does NOT nullify the observed layer. A
yeast experiment whose process is not in the registry still produces
`observed.organism = "Saccharomyces cerevisiae"` so downstream agents
have something to reason from.

Decision flow (single LLM call):

    manifest provided?  -> both layers from manifest, provenance=MANIFEST
    narrative empty?    -> both layers UNKNOWN, no LLM call
    LLM error/timeout   -> both layers UNKNOWN
    LLM success         -> validate observed and registered independently
                           (each may succeed or fall to UNKNOWN on its own)

The extractor never raises. All failure paths return UNKNOWN with a
rationale.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from fermdocs.domain.models import (
    EvidenceLocator,
    IdentityProvenance,
    NarrativeBlock,
    ObservedFacts,
    ProcessIdentity,
    RegisteredProcess,
    ScaleInfo,
)
from fermdocs.mapping.evidence_gated_llm import (
    LLM_CONFIDENCE_CAP,
    verify_substring_evidence,
)
from fermdocs.mapping.process_registry import (
    ProcessRegistry,
    fingerprint_check,
)


_SYSTEM_PROMPT = (
    "You classify the process described in a fermentation experiment.\n\n"
    "You will be given:\n"
    "  - a CLOSED LIST of allowed registry process IDs with their organism,\n"
    "    product, and known aliases\n"
    "  - prose paragraphs from the experiment's source documents\n\n"
    "Emit a JSON object with TWO sub-objects:\n\n"
    "1. observed: the surface facts you can pull directly from prose.\n"
    "   Keys: organism, product, process_family_hint, scale_volume_l,\n"
    "         vessel_type, confidence, evidence (list of"
    " {paragraph_idx, span_text}), rationale.\n"
    "   Each populated text field MUST be backed by an evidence span that\n"
    "   is a verbatim substring of one of the input paragraphs.\n"
    "   If you cannot find a field in the prose, set it to null.\n\n"
    "2. registered: the registry classification.\n"
    "   Keys: process_id (string|null - MUST be from the listed registry"
    " IDs or null), confidence, rationale.\n"
    "   Only emit a process_id if you are confident it matches one of the\n"
    "   listed entries. If unsure, use null.\n\n"
    "Rules:\n"
    "  - You may ONLY emit observed fields whose values appear (verbatim)\n"
    "    in the input paragraphs. Do not invent values.\n"
    "  - You may ONLY emit a process_id from the provided registry list.\n"
    "  - Confidence is your subjective confidence (0..1). It will be\n"
    "    capped at 0.85 downstream regardless of what you emit.\n"
    "  - The two layers are independent: a paper that names yeast but is\n"
    "    not in the registry should populate observed and leave\n"
    "    registered.process_id null."
)


class IdentityLLMClient(Protocol):
    """Minimal protocol so tests can supply a scripted client."""

    def call(self, system: str, user: str) -> dict[str, Any]: ...


class IdentityExtractor:
    """Wraps an IdentityLLMClient with the two-layer safety stack."""

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
                observed=ObservedFacts(
                    provenance=IdentityProvenance.UNKNOWN,
                    rationale="no narrative blocks; LLM not called",
                ),
                registered=RegisteredProcess(
                    provenance=IdentityProvenance.UNKNOWN,
                    rationale="no narrative blocks; LLM not called",
                ),
            )

        if self._client is None:
            return ProcessIdentity(
                observed=ObservedFacts(
                    provenance=IdentityProvenance.UNKNOWN,
                    rationale="no LLM client configured",
                ),
                registered=RegisteredProcess(
                    provenance=IdentityProvenance.UNKNOWN,
                    rationale="no LLM client configured",
                ),
            )

        try:
            payload = self._client.call(
                _SYSTEM_PROMPT,
                _render_user_prompt(self._registry, narrative_blocks),
            )
        except Exception as exc:
            err = f"LLM call failed: {exc.__class__.__name__}"
            return ProcessIdentity(
                observed=ObservedFacts(
                    provenance=IdentityProvenance.UNKNOWN, rationale=err
                ),
                registered=RegisteredProcess(
                    provenance=IdentityProvenance.UNKNOWN, rationale=err
                ),
            )

        observed = _validate_observed(
            payload.get("observed") or {},
            narrative_blocks=narrative_blocks,
            default_file_id=file_id,
        )
        registered = _validate_registered(
            payload.get("registered") or {},
            registry=self._registry,
            present_variables=present_variables,
        )
        return ProcessIdentity(observed=observed, registered=registered)


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
        f"REGISTRY (allowed process IDs):\n{process_block}\n\n"
        f"PARAGRAPHS:\n{json.dumps(paragraphs, ensure_ascii=False)}\n\n"
        "Return JSON with two top-level keys: 'observed' and 'registered'."
        " The observed layer carries surface facts pulled from prose."
        " The registered layer carries a process_id from the registry list (or null)."
    )


def _validate_observed(
    raw: dict[str, Any],
    *,
    narrative_blocks: list[NarrativeBlock],
    default_file_id: str | None,
) -> ObservedFacts:
    """Validate the observed layer field-by-field.

    For each populated text field, the LLM must cite an evidence span that
    is a verbatim substring of one of the input paragraphs. A field whose
    evidence fails verification is nulled (kept fields survive). If no
    fields survive, the layer is UNKNOWN.
    """
    by_paragraph = {
        b.locator.get("paragraph_idx", -1): b for b in narrative_blocks
    }

    raw_evidence = raw.get("evidence") or []
    locators: list[EvidenceLocator] = []
    paragraph_text_by_idx: dict[int, str] = {
        idx: b.text for idx, b in by_paragraph.items()
    }
    failed_reasons: list[str] = []

    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        para_idx = item.get("paragraph_idx")
        span = item.get("span_text") or ""
        if para_idx not in by_paragraph:
            failed_reasons.append(f"unknown paragraph_idx {para_idx}")
            continue
        block = by_paragraph[para_idx]
        ok, reason = verify_substring_evidence(span, block.text, value=None)
        if not ok:
            failed_reasons.append(reason or "evidence rejected")
            continue
        locators.append(
            EvidenceLocator(
                file_id=block.locator.get("file_id") or default_file_id or "",
                paragraph_idx=int(para_idx),
                span_text=span,
                span_start=block.text.find(span) if span else None,
            )
        )

    # A field is "supported" if it appears verbatim in at least one
    # successfully-verified evidence span. Unsupported fields get nulled.
    supported_text = " ".join(loc.span_text for loc in locators)

    def _supported(value: str | None) -> bool:
        if not value:
            return False
        return value in supported_text

    organism = raw.get("organism")
    product = raw.get("product")
    family_hint = raw.get("process_family_hint")

    organism_kept = organism if _supported(organism) else None
    product_kept = product if _supported(product) else None
    family_kept = family_hint if _supported(family_hint) else None

    # Scale fields are numeric/categorical; we accept them if any evidence
    # passed verification (they're typically named in the same span as
    # organism/product). Tighter checks possible later.
    scale_volume = raw.get("scale_volume_l")
    vessel_type = raw.get("vessel_type")
    scale: ScaleInfo | None = None
    if locators and (scale_volume is not None or vessel_type):
        try:
            vol = float(scale_volume) if scale_volume is not None else None
        except (TypeError, ValueError):
            vol = None
        scale = ScaleInfo(
            volume_l=vol, vessel_type=vessel_type if vessel_type else None
        )

    raw_conf = raw.get("confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(LLM_CONFIDENCE_CAP, confidence))

    has_any_field = any([organism_kept, product_kept, family_kept, scale])
    if not has_any_field:
        # Always lead with the evidence-check verdict; preserve LLM's own
        # rationale as a trailing note. Downstream agents and fixtures look
        # for the word "evidence" to confirm the rejection mode.
        verdict = "no observed fields survived evidence check"
        if failed_reasons:
            verdict = f"{verdict}: {failed_reasons}"
        else:
            # Fields were emitted but none appeared inside any verified
            # span. That's still an evidence-check failure; say so.
            verdict = f"{verdict}: emitted fields not present in any verified evidence span"
        llm_rationale = raw.get("rationale")
        rationale = (
            f"{verdict}; LLM said: {llm_rationale}" if llm_rationale else verdict
        )
        return ObservedFacts(
            provenance=IdentityProvenance.UNKNOWN,
            rationale=rationale,
            evidence_locators=locators[:5],
        )

    rationale_parts = []
    if raw.get("rationale"):
        rationale_parts.append(str(raw["rationale"]))
    if failed_reasons:
        rationale_parts.append(f"some fields nulled: {failed_reasons}")
    rationale = "; ".join(rationale_parts) if rationale_parts else None

    return ObservedFacts(
        organism=organism_kept,
        product=product_kept,
        process_family_hint=family_kept,
        scale=scale,
        confidence=confidence,
        provenance=IdentityProvenance.LLM_WHITELISTED,
        evidence_locators=locators[:5],
        rationale=rationale,
    )


def _validate_registered(
    raw: dict[str, Any],
    *,
    registry: ProcessRegistry,
    present_variables: set[str],
) -> RegisteredProcess:
    process_id = raw.get("process_id")

    if not process_id:
        return RegisteredProcess(
            provenance=IdentityProvenance.UNKNOWN,
            rationale=str(raw.get("rationale") or "LLM returned null process_id"),
        )

    by_id = registry.by_id()
    if process_id not in by_id:
        return RegisteredProcess(
            provenance=IdentityProvenance.UNKNOWN,
            rationale=f"off-whitelist process_id {process_id!r}",
        )

    entry = by_id[process_id]
    ok, reason = fingerprint_check(entry, present_variables)
    if not ok:
        return RegisteredProcess(
            process_id=None,
            provenance=IdentityProvenance.UNKNOWN,
            rationale=f"fingerprint mismatch for {process_id!r}: {reason}",
        )

    raw_conf = raw.get("confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(LLM_CONFIDENCE_CAP, confidence))

    return RegisteredProcess(
        process_id=process_id,
        confidence=confidence,
        provenance=IdentityProvenance.LLM_WHITELISTED,
        rationale=raw.get("rationale"),
    )
