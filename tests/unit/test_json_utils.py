"""loads_lenient: recover LLM JSON that strict json.loads rejects.

Regression: a characterize/diagnose response containing an invalid backslash
escape (chemical name / unit / math) crashed a whole stage with
'Invalid \\uXXXX escape'. The lenient parser repairs it.
"""

from __future__ import annotations

import json

import pytest

from fermdocs.json_utils import loads_lenient


def test_valid_json_passes_through():
    assert loads_lenient('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_invalid_u_escape_recovered():
    # \uXYZ is not a valid \uXXXX escape -> strict json.loads raises.
    assert loads_lenient(r'{"v": "MnSO4 \uXYZ"}') == {"v": r"MnSO4 \uXYZ"}


def test_lone_backslash_and_math_escapes_recovered():
    assert loads_lenient(r'{"n": "rate \mu and c:\path"}') == {"n": r"rate \mu and c:\path"}


def test_code_fence_stripped():
    assert loads_lenient("```json\n{\"x\": 2}\n```") == {"x": 2}


def test_double_encoded_unwrapped():
    assert loads_lenient(json.dumps(json.dumps({"w": True}))) == {"w": True}


def test_unrecoverable_still_raises():
    with pytest.raises(json.JSONDecodeError):
        loads_lenient("not json at all {{{")
