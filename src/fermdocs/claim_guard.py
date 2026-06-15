"""Claim guard: reject agent claims that contradict the deterministic facts.

The deterministic metrics can be correct and an LLM agent can still assert the
opposite in free text — the praaj review caught four such classes:

  * "no DO data available" / "no substrate limitation metrics" when those
    channels ARE populated (false unavailability).
  * "DO = 0 = oxygen bottleneck" on a process operating anaerobically (DO pinned
    near zero for most of the run — the regime, not a transfer limitation).
  * "scale / bioreactor confound" when every run is the SAME reactor and the
    differing number is the initial CHARGE volume, not hardware scale.
  * "growth rate at t=0" — a rate needs an interval; t=0 is a single point.

This module is a deterministic, defense-in-depth guard over claim TEXT. It is
pure (no I/O, no LLM): `check_claim(text, facts)` returns the violations, and the
caller decides whether to reject, downgrade, or annotate. Facts are data-derived
(`ClaimFacts`); nothing here encodes a nominal/spec value — only the vocabulary
needed to recognise which channel a sentence is talking about.

Conservative by design: a check fires only on an explicit contradicting pattern
(negation for availability; anaerobic_operation for oxygen; known-constant scale
for the scale confound), so legitimate claims pass untouched.
"""
from __future__ import annotations

import re
import statistics as _stats
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimFacts:
    """Deterministic facts a claim must not contradict. All data-derived."""

    populated_channels: frozenset[str] = frozenset()  # variables present + populated
    anaerobic_operation: bool = False     # DO pinned near zero for most of the run
    reactor_scale_constant: bool | None = None  # None = unknown (don't flag scale)
    sampling_resolution_h: float | None = None


@dataclass(frozen=True)
class ClaimViolation:
    code: str       # machine code, e.g. "false_unavailability"
    message: str    # the deterministic correction, for the caller to surface


# Vocabulary only (synonyms -> the channel-name token they refer to). This is
# naming, NOT a nominal value: it lets the guard tell which populated channel a
# sentence denies. Extend freely; unknown phrases simply won't match a channel.
_CHANNEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "do": ("dissolved oxygen", "dissolved o2", "do", "do%", "po2", "oxygen"),
    "substrate": ("substrate", "sugar", "glucose", "sucrose", "carbon source"),
    "product": ("product", "titer", "titre", "lactic acid", "lactate"),
    "biomass": ("biomass", "cell mass", "dcw", "od", "od600", "wcw"),
    "ph": ("ph",),
    "volume": ("volume", "broth volume", "working volume"),
    "temperature": ("temperature", "temp"),
}

# Negation patterns that ASSERT a metric/channel is unavailable. The captured
# group is the thing claimed missing; we then test it against the channel set.
_UNAVAIL_PATTERNS = [
    re.compile(r"\bno\s+([a-z0-9 /%_-]{1,40}?)\s+(?:data|measurements?|metrics?|"
               r"trajector\w*|readings?|values?|information)\b", re.I),
    re.compile(r"\bno\s+(?:data|measurements?|metrics?|information)\s+(?:on|for|about)\s+"
               r"([a-z0-9 /%_-]{1,40})", re.I),
    re.compile(r"\b([a-z0-9 /%_-]{1,40}?)\s+(?:is|are|was|were)?\s*(?:not available|"
               r"unavailable|not present|not measured|not reported|missing)\b", re.I),
    re.compile(r"\b(?:lack|absence)\s+of\s+([a-z0-9 /%_-]{1,40}?)\s+(?:data|metrics?|measurements?)",
               re.I),
    re.compile(r"\b(?:without|absent)\s+([a-z0-9 /%_-]{1,40}?)\s+(?:data|metrics?|measurements?)",
               re.I),
]

_OXYGEN_LIMIT_RE = re.compile(
    r"\b(oxygen[\s-]*limit\w*|o2[\s-]*limit\w*|o2[\s-]*transfer[\s-]*limit\w*|"
    r"do\s*bottleneck|oxygen\s*bottleneck|mass[\s-]*transfer\s*(?:failure|limit\w*|bottleneck)|"
    r"oxygen[\s-]*starv\w*|oxygen[\s-]*deficien\w*|insufficient\s*(?:aeration|oxygen))\b",
    re.I)

_SCALE_CONFOUND_RE = re.compile(
    r"\b(scale\s*confound|scale\s*difference|different\s*scale|scale[\s-]*up\s*confound|"
    r"bioreactor\s*confound|different\s*(?:bio)?reactor|reactor\s*(?:size|scale)\s*"
    r"(?:difference|confound|change)|scale[\s-]*dependent\s*confound)\b",
    re.I)

# A rate word adjacent to "t=0" / "time 0" / "t0". A rate is interval-based.
_RATE_AT_T0_RE = re.compile(
    r"\b(?:rate|μ|mu|mu_max|μ_max|specific\s+growth|growth\s+rate)\b[^.]{0,40}?"
    r"\bat\s*(?:t\s*=?\s*0|time\s*(?:=\s*)?0|t0|zero\s*(?:hours?|h)?)\b",
    re.I)


_NEGATION = re.compile(r"\b(not|no|never|without|isn'?t|aren'?t|wasn'?t|weren'?t|"
                        r"consistent with anaerobic|rather than)\b", re.I)


def _negated_before(text: str, start: int, *, window: int = 28) -> bool:
    """True if a negation token appears just before `start` — so a phrase like
    'NOT an oxygen limitation' or 'consistent with anaerobic, not a bottleneck'
    is read as already-corrected and does not fire a violation."""
    return bool(_NEGATION.search(text[max(0, start - window): start]))


def _channel_tokens(channel: str) -> set[str]:
    """Searchable phrases for a channel: its name stem (minus unit suffixes) plus
    known synonyms. e.g. 'do_pct_saturation' -> {'do', 'dissolved oxygen', ...}."""
    name = channel.lower()
    stem = name
    for suffix in ("_g_l", "_mg_l", "_pct_saturation", "_pct", "_au", "_l", "_c",
                   "_per_l", "_ratio", "_frac"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    tokens = {stem.replace("_", " "), stem}
    for key, syns in _CHANNEL_SYNONYMS.items():
        if key in stem or stem in key:
            tokens.update(syns)
    return {t for t in tokens if len(t) >= 2}


def check_claim(text: str, facts: ClaimFacts) -> list[ClaimViolation]:
    """Return the deterministic-fact contradictions in `text` (may be empty)."""
    if not text:
        return []
    low = text.lower()
    out: list[ClaimViolation] = []

    # 1. False unavailability: a negation naming a channel that IS populated.
    if facts.populated_channels:
        chan_tokens = {c: _channel_tokens(c) for c in facts.populated_channels}
        for pat in _UNAVAIL_PATTERNS:
            for m in pat.finditer(low):
                denied = m.group(1).strip()
                for chan, toks in chan_tokens.items():
                    if any(tok and tok in denied for tok in toks):
                        out.append(ClaimViolation(
                            "false_unavailability",
                            f"Claim asserts '{denied}' is unavailable, but channel "
                            f"'{chan}' IS present and populated. Rejected: check the "
                            "channel registry before asserting a metric is missing."))
                        break

    # 2. Oxygen limitation on an anaerobically-operating process. Skip
    #    negated/already-corrected phrasing ("NOT an oxygen limitation").
    if facts.anaerobic_operation:
        m = _OXYGEN_LIMIT_RE.search(low)
        if m and not _negated_before(low, m.start()):
            out.append(ClaimViolation(
                "oxygen_limitation_when_anaerobic",
                "Claim invokes oxygen/DO limitation, but DO was pinned near zero for "
                "most of the run (anaerobic operation): DO=0 is the operating regime "
                "here, not a transfer limitation. A real aerobic O2 limitation shows "
                "transient dips or a controlled positive setpoint. Reframe as "
                "'consistent with anaerobic/microaerophilic operation'."))

    # 3. Scale confound when the reactor scale is constant per metadata.
    if facts.reactor_scale_constant:
        m = _SCALE_CONFOUND_RE.search(low)
        if m and not _negated_before(low, m.start()):
            out.append(ClaimViolation(
                "scale_confound_when_constant",
                "Claim invokes a reactor scale/bioreactor confound, but reactor scale "
                "is constant across runs per metadata. A differing time-series volume "
                "is the initial CHARGE (a process variable), not a hardware scale "
                "difference. Rejected."))

    # 4. A rate reported at t=0 (single point cannot yield a rate).
    if _RATE_AT_T0_RE.search(low):
        res = facts.sampling_resolution_h
        span = (f" first resolvable interval is 0–{res:.0f}h" if res else
                " report the first interval, not t=0")
        out.append(ClaimViolation(
            "rate_at_t0",
            "Claim reports a rate 'at t=0', but a rate needs two timepoints;"
            f"{span}. Restate as a first-interval rate and declare the resolution."))

    return out


# -----------------------------------------------------------------------------
# Fact assembly from a bundle's own metric findings + metadata (duck-typed so it
# works on any object exposing `.statistics` (dict) and `.summary`).
# -----------------------------------------------------------------------------

def _stat(f, key):
    return (getattr(f, "statistics", None) or {}).get(key)


def anaerobic_operation_from_findings(findings) -> bool:
    """True iff DO was measured (A14 ran) and the runs operate anaerobically (DO
    pinned near zero for most of the run) — the data signal that O2-limitation
    talk is wrong. Catches the saturated-at-t0-then-crashes pattern."""
    seen = [_stat(f, "anaerobic_operation") for f in findings
            if _stat(f, "metric_id") == "A14" and "anaerobic_operation" in (getattr(f, "statistics", None) or {})]
    return bool(seen) and all(bool(v) for v in seen)


def sampling_resolution_from_findings(findings) -> float | None:
    """Median sampling resolution across the rate/phase findings that carry it."""
    vals = [_stat(f, "sampling_resolution_h") for f in findings
            if isinstance(_stat(f, "sampling_resolution_h"), (int, float))]
    return float(_stats.median(vals)) if vals else None


def reactor_scale_constant(dossier) -> bool | None:
    """True iff metadata pins a single reactor working volume (so a run-to-run
    volume difference is the initial CHARGE, not a hardware scale change). None
    when scale metadata is absent — then scale claims are not flagged."""
    if not isinstance(dossier, dict):
        return None
    for loc in (dossier.get("scale"), (dossier.get("identity") or {}).get("scale"),
                (dossier.get("registered_process") or {}).get("scale")):
        if isinstance(loc, dict) and loc.get("volume_l") is not None:
            return True
    if dossier.get("scale_volume_l") is not None:
        return True
    return None


def build_claim_facts(findings, populated_channels, dossier=None) -> ClaimFacts:
    """Assemble the deterministic facts a claim must not contradict, from the
    bundle's own metrics + metadata (all data-derived)."""
    return ClaimFacts(
        populated_channels=frozenset(populated_channels),
        anaerobic_operation=anaerobic_operation_from_findings(findings),
        reactor_scale_constant=reactor_scale_constant(dossier),
        sampling_resolution_h=sampling_resolution_from_findings(findings),
    )
