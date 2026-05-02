"""Shared safety primitive for LLM calls that emit evidence-cited claims.

Multiple extractors in this codebase need the same safety stack:
  - confidence cap so LLM outputs never auto-accept
  - substring-evidence verification (hallucination guard)
  - graceful error degradation (return None, never raise)

The narrative extractor (Tier 2 numeric extraction from prose) and the
identity extractor (process-identity classification from prose) are both
LLM-driven, both must cite evidence, and both must fail soft. Rather than
copying the safety logic, both call into this module.

Public surface:
  - LLM_CONFIDENCE_CAP, MAX_EVIDENCE_LEN, MAX_SENTENCE_BREAKS
  - verify_substring_evidence(evidence, source_text, value)
  - value_string_forms(value)

Backwards-compat:
  - narrative_extractor.py re-exports verify_evidence as an alias for
    verify_substring_evidence so existing imports keep working.
"""

from __future__ import annotations

from typing import Any

LLM_CONFIDENCE_CAP = 0.85
MAX_EVIDENCE_LEN = 200
MAX_SENTENCE_BREAKS = 2


def verify_substring_evidence(
    evidence: str, source_text: str, value: Any
) -> tuple[bool, str | None]:
    """Hallucination guard. Returns (ok, reason_if_rejected).

    Rules:
      1. Evidence must be non-empty and short (<= MAX_EVIDENCE_LEN chars).
      2. Evidence must be a verbatim substring of source_text.
      3. Value's string form must appear inside the evidence (not just the source).
      4. Evidence must span <= MAX_SENTENCE_BREAKS sentence terminators.

    `value=None` skips rule 3 entirely (used by extractors where the cited
    field is not numeric, e.g. an organism name).
    """
    if not evidence:
        return False, "evidence empty"
    if len(evidence) > MAX_EVIDENCE_LEN:
        return False, f"evidence too long ({len(evidence)} chars)"
    if evidence not in source_text:
        return False, "evidence not in source"
    if value is not None:
        candidates = value_string_forms(value)
        if candidates and not any(c in evidence for c in candidates if c):
            return False, f"value {value!r} not within evidence span"
    breaks = sum(1 for c in evidence if c in ".!?")
    if breaks > MAX_SENTENCE_BREAKS:
        return False, f"evidence spans too many sentences ({breaks})"
    return True, None


def value_string_forms(value: Any) -> set[str]:
    """Common rendering variants of a value. Any of these inside evidence counts as a match."""
    forms: set[str] = set()
    if value is None:
        return forms
    forms.add(str(value).strip())
    try:
        f = float(value)
        forms.add(str(f))
        if f == int(f):
            forms.add(str(int(f)))
        forms.add(f"{f:.1f}")
        forms.add(f"{f:.2f}")
        forms.add(f"{f:.3f}")
    except (TypeError, ValueError):
        pass
    return {f for f in forms if f}
