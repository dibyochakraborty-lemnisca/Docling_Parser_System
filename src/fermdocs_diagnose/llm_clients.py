"""Concrete DiagnosisLLMClient implementations for Gemini and Anthropic.

The diagnosis ReAct loop sends a running `messages` list and expects a single
JSON dict per turn with `action: tool_call|emit`. Both providers use
structured-output mode so the wire format is locked: the agent never has to
parse free-form text.

Schema is enforced at the API boundary, not at parse time. Hallucination
guards (citation integrity, provenance downgrade, forbidden-phrase warnings)
live in `validators.py` and run after every emit.
"""

from __future__ import annotations

import json
import os
from typing import Any

_GEMINI_DEFAULT_MODEL = "gemini-3-pro"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-7"


class GeminiDiagnosisClient:
    """Implements DiagnosisLLMClient via Google Gemini structured output."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_DIAGNOSIS_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        # Gemini wants `contents` as a list of {role, parts} dicts. Map our
        # OpenAI-flavored {role, content} into that shape; user/assistant
        # roles map to "user"/"model".
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_GEMINI_DIAGNOSIS_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_DIAGNOSIS"):
            import sys

            print(f"[gemini-diagnosis] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini returned empty diagnosis response")
        return json.loads(text)


class AnthropicDiagnosisClient:
    """Implements DiagnosisLLMClient via Anthropic tool-use.

    Two tools: `tool_call` (the agent wants to fetch more data) and `emit`
    (the agent has reached its conclusion). tool_choice='any' lets the
    model pick which to invoke per turn.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_DIAGNOSIS_MODEL")
            or _ANTHROPIC_DEFAULT_MODEL
        )

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=[
                {
                    "name": "tool_call",
                    "description": "Request more data from a diagnosis tool.",
                    "input_schema": _ANTHROPIC_TOOL_CALL_SCHEMA,
                },
                {
                    "name": "emit",
                    "description": (
                        "Emit final diagnosis claims and conclude reasoning."
                    ),
                    "input_schema": _ANTHROPIC_EMIT_SCHEMA,
                },
            ],
            tool_choice={"type": "any"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = dict(block.input)
                # Mirror the tool name into the action discriminator the
                # agent expects, so both providers feed agent.py the same
                # shape.
                payload["action"] = block.name
                return payload
        raise ValueError("anthropic diagnosis response missing tool_use block")


def build_diagnosis_client(provider: str | None = None):
    """Pick a DiagnosisLLMClient by provider name.

    Resolution: explicit arg > FERMDOCS_DIAGNOSIS_PROVIDER >
    FERMDOCS_MAPPER_PROVIDER > 'gemini'. 'none' / 'fake' returns None so
    the agent enters error-output mode (used by tests).
    """
    name = (
        provider
        or os.environ.get("FERMDOCS_DIAGNOSIS_PROVIDER")
        or os.environ.get("FERMDOCS_MAPPER_PROVIDER", "gemini")
    ).lower()
    if name in ("fake", "none"):
        return None
    if name == "gemini":
        return GeminiDiagnosisClient()
    if name == "anthropic":
        return AnthropicDiagnosisClient()
    raise ValueError(
        f"unknown diagnosis provider: {name!r} "
        "(expected 'anthropic', 'gemini', 'fake', or 'none')"
    )


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


_GEMINI_TOOL_CALL_FIELDS: dict[str, Any] = {
    "tool": {
        "type": "STRING",
        "enum": [
            # Wave 1 surface
            "get_finding",
            "get_trajectory",
            "get_spec",
            # Stage 2 / 3 bundle surface
            "list_runs",
            "get_meta",
            "get_findings",
            "get_specs",
            "get_timecourse",
            "execute_python",
            "submit_diagnosis",
            # Plan A Stage 2: organism-aware priors
            "get_priors",
            # Plan B Stage 3: prose insights from source document
            "get_narrative_observations",
        ],
        "nullable": True,
    },
    # Gemini structured output requires explicit properties on args. We
    # union every tool's possible arg keys here. The dispatcher already
    # passes args via **kwargs, so unused keys are tolerated by the tool
    # methods. Fields are nullable so the model can leave them out.
    "args": {
        "type": "OBJECT",
        "nullable": True,
        "properties": {
            # execute_python
            "code": {"type": "STRING", "nullable": True},
            "timeout": {"type": "INTEGER", "nullable": True},
            # get_finding(s) / filtering
            "finding_id": {"type": "STRING", "nullable": True},
            "run_id": {"type": "STRING", "nullable": True},
            "variable": {"type": "STRING", "nullable": True},
            "severity": {"type": "STRING", "nullable": True},
            "tier": {"type": "STRING", "nullable": True},
            "limit": {"type": "INTEGER", "nullable": True},
            "max_points": {"type": "INTEGER", "nullable": True},
            "time_range_h": {
                "type": "ARRAY",
                "items": {"type": "NUMBER"},
                "nullable": True,
            },
            # get_priors
            "organism": {"type": "STRING", "nullable": True},
            "process_family": {"type": "STRING", "nullable": True},
            # get_narrative_observations
            "tag": {"type": "STRING", "nullable": True},
            # submit_diagnosis carries an opaque payload — Gemini structured
            # output can't model arbitrary recursive shapes, so we provide a
            # stringified payload escape hatch and parse it on receipt.
            "payload_json": {"type": "STRING", "nullable": True},
        },
    },
}

_GEMINI_CLAIM_BASE_FIELDS: dict[str, Any] = {
    "summary": {"type": "STRING"},
    "cited_finding_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
    "cited_narrative_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
    "affected_variables": {"type": "ARRAY", "items": {"type": "STRING"}},
    "confidence": {"type": "NUMBER"},
    "confidence_basis": {
        "type": "STRING",
        "enum": ["schema_only", "process_priors", "cross_run"],
    },
    "domain_tags": {"type": "ARRAY", "items": {"type": "STRING"}},
}

_GEMINI_DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["tool_call", "emit"]},
        # tool_call branch
        **_GEMINI_TOOL_CALL_FIELDS,
        # emit branch
        "failures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    **_GEMINI_CLAIM_BASE_FIELDS,
                    "severity": {
                        "type": "STRING",
                        "enum": ["info", "minor", "major", "critical"],
                    },
                    "cited_trajectories": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "run_id": {"type": "STRING"},
                                "variable": {"type": "STRING"},
                            },
                            "required": ["run_id", "variable"],
                        },
                    },
                    "time_window": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER", "nullable": True},
                            "end": {"type": "NUMBER", "nullable": True},
                        },
                        "nullable": True,
                    },
                },
            },
        },
        "trends": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    **_GEMINI_CLAIM_BASE_FIELDS,
                    "direction": {
                        "type": "STRING",
                        "enum": [
                            "increasing",
                            "decreasing",
                            "plateau",
                            "oscillating",
                        ],
                    },
                    "cited_trajectories": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "run_id": {"type": "STRING"},
                                "variable": {"type": "STRING"},
                            },
                            "required": ["run_id", "variable"],
                        },
                    },
                    "time_window": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER", "nullable": True},
                            "end": {"type": "NUMBER", "nullable": True},
                        },
                        "nullable": True,
                    },
                },
            },
        },
        "analysis": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    **_GEMINI_CLAIM_BASE_FIELDS,
                    "kind": {
                        "type": "STRING",
                        "enum": [
                            "cross_run_observation",
                            "data_quality_caveat",
                            "spec_alignment",
                            "phase_characterization",
                        ],
                    },
                },
            },
        },
        "open_questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "why_it_matters": {"type": "STRING"},
                    "cited_finding_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "cited_narrative_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "answer_format_hint": {
                        "type": "STRING",
                        "enum": ["yes_no", "free_text", "numeric", "categorical"],
                    },
                    "domain_tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
            },
        },
        "narrative": {"type": "STRING", "nullable": True},
    },
    "required": ["action"],
}


_ANTHROPIC_TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                # Wave 1 surface
                "get_finding",
                "get_trajectory",
                "get_spec",
                # Stage 2 / 3 bundle surface
                "list_runs",
                "get_meta",
                "get_findings",
                "get_specs",
                "get_timecourse",
                "execute_python",
                "submit_diagnosis",
                # Plan A Stage 2: organism-aware priors
                "get_priors",
                # Plan B Stage 3: prose insights from source document
                "get_narrative_observations",
            ],
        },
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

_ANTHROPIC_CLAIM_BASE: dict[str, Any] = {
    "summary": {"type": "string"},
    "cited_finding_ids": {"type": "array", "items": {"type": "string"}},
    "cited_narrative_ids": {"type": "array", "items": {"type": "string"}},
    "affected_variables": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number"},
    "confidence_basis": {
        "type": "string",
        "enum": ["schema_only", "process_priors", "cross_run"],
    },
    "domain_tags": {"type": "array", "items": {"type": "string"}},
}

_ANTHROPIC_EMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_ANTHROPIC_CLAIM_BASE,
                    "severity": {
                        "type": "string",
                        "enum": ["info", "minor", "major", "critical"],
                    },
                    "cited_trajectories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "run_id": {"type": "string"},
                                "variable": {"type": "string"},
                            },
                            "required": ["run_id", "variable"],
                        },
                    },
                },
            },
        },
        "trends": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_ANTHROPIC_CLAIM_BASE,
                    "direction": {
                        "type": "string",
                        "enum": [
                            "increasing",
                            "decreasing",
                            "plateau",
                            "oscillating",
                        ],
                    },
                    "cited_trajectories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "run_id": {"type": "string"},
                                "variable": {"type": "string"},
                            },
                            "required": ["run_id", "variable"],
                        },
                    },
                },
            },
        },
        "analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_ANTHROPIC_CLAIM_BASE,
                    "kind": {
                        "type": "string",
                        "enum": [
                            "cross_run_observation",
                            "data_quality_caveat",
                            "spec_alignment",
                            "phase_characterization",
                        ],
                    },
                },
            },
        },
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "cited_finding_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cited_narrative_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "answer_format_hint": {
                        "type": "string",
                        "enum": [
                            "yes_no",
                            "free_text",
                            "numeric",
                            "categorical",
                        ],
                    },
                    "domain_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "narrative": {"type": ["string", "null"]},
    },
}
