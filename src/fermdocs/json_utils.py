"""Lenient JSON parsing for LLM structured-output responses.

Gemini/Anthropic occasionally emit JSON that ``json.loads`` rejects:
code fences, surrounding prose, a double-encoded JSON string, or invalid
backslash escapes (e.g. ``\\mu``, ``\\alpha``, or ``\\u`` not followed by 4 hex
digits). The last one is common when the text contains chemical names, math,
units, or file-path-like tokens, and a single stray escape otherwise crashes a
whole stage with a cryptic ``Invalid \\uXXXX escape`` at a deep char offset.

``loads_lenient`` repairs those before parsing. The repair only fires after a
strict parse fails, so well-formed responses are untouched.
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


def loads_lenient(text: str):
    """Parse LLM JSON, tolerating code fences, surrounding prose, double
    encoding, and invalid backslash escapes. Raises json.JSONDecodeError only
    if every recovery attempt fails."""
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
        return _unwrap(json.loads(_BAD_ESCAPE.sub(r"\\\\", t)))
