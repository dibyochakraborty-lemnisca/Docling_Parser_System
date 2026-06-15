"""Lenient JSON parsing for LLM structured-output responses.

Gemini/Anthropic occasionally emit JSON that ``json.loads`` rejects:
code fences, surrounding prose, a double-encoded JSON string, or invalid
backslash escapes (e.g. ``\\mu``, ``\\alpha``, or ``\\u`` not followed by 4 hex
digits). The last one is common when the text contains chemical names, math,
units, or file-path-like tokens, and a single stray escape otherwise crashes a
whole stage with a cryptic ``Invalid \\uXXXX escape`` at a deep char offset.

``loads_lenient`` repairs those before parsing. The repair only fires after a
strict parse fails, so well-formed responses are untouched.

It also SALVAGES truncated JSON. A thinking model can spend its whole output
budget on reasoning and return the visible JSON cut off mid-string (a "char N
unterminated string" error). That single truncated reply must not crash a whole
multi-agent stage, so the salvage rebuilds the longest valid prefix — closing the
open string/brackets and dropping the incomplete trailing field — yielding a
PARTIAL object the caller can read with ``.get`` instead of an exception.
"""

from __future__ import annotations

import json
import re

# A backslash that does NOT begin a valid JSON escape (" \ / b f n r t, or
# u+4hex). Doubling it makes the string valid JSON without changing meaning.
_BAD_ESCAPE = re.compile(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})')


def _unwrap(obj):
    """Gemini sometimes double-encodes: a JSON *string* whose content is JSON.
    Decode one more layer so callers always get the structure they expect."""
    return json.loads(obj) if isinstance(obj, str) else obj


def _salvage_truncated(t: str):
    """Recover a PARTIAL object/array from JSON truncated mid-output.

    Walks the text tracking string state + bracket nesting, remembering the last
    position where a complete VALUE (not a key) closed at the current depth.
    Cuts there, drops a dangling comma, and appends the closers for whatever is
    still open. Returns the parsed partial structure, or None if it can't form
    valid JSON — so a caller only ever sees a dict it can read or a clean failure.
    """
    start = t.find("{")
    sb = t.find("[")
    if sb != -1 and (start == -1 or sb < start):
        start = sb
    if start == -1:
        return None
    t = t[start:]
    n = len(t)
    stack: list[str] = []
    in_str = esc = False
    best: tuple[int, str] | None = None  # (cut index exclusive, closing string)

    def _closers() -> str:
        return "".join("}" if c == "{" else "]" for c in reversed(stack))

    i = 0
    while i < n:
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                # A string just closed. It's a complete value only if it isn't a
                # key (i.e. not followed by ':'). Keys must not be a cut point.
                j = i + 1
                while j < n and t[j] in " \t\r\n":
                    j += 1
                if not (j < n and t[j] == ":"):
                    best = (i + 1, _closers())
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            best = (i + 1, _closers())  # a container completed = complete value
        elif ch == ",":
            best = (i, _closers())       # everything before the comma is complete
        i += 1

    if best is None:
        return None
    cut, close = best
    candidate = t[:cut].rstrip().rstrip(",") + close
    try:
        return _unwrap(json.loads(candidate))
    except json.JSONDecodeError:
        return None


def loads_lenient(text: str):
    """Parse LLM JSON, tolerating code fences, surrounding prose, double
    encoding, invalid backslash escapes, and truncated output (salvaged to a
    partial object). Raises json.JSONDecodeError only if every recovery attempt
    fails."""
    t = (text or "").strip()
    if "```" in t:
        t = t.split("```", 2)[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    try:
        return _unwrap(json.loads(t))
    except json.JSONDecodeError:
        pass
    # Carve out the outermost {...} (prose-wrapped JSON), then tolerate stray
    # backslash escapes.
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    try:
        return _unwrap(json.loads(t))
    except json.JSONDecodeError:
        pass
    repaired = _BAD_ESCAPE.sub(r"\\\\", t)
    try:
        return _unwrap(json.loads(repaired))
    except json.JSONDecodeError:
        # Last resort: truncated mid-output -> salvage the longest valid prefix
        # rather than crash the whole stage on one cut-off reply.
        salvaged = _salvage_truncated(repaired)
        if salvaged is not None:
            return salvaged
        raise
