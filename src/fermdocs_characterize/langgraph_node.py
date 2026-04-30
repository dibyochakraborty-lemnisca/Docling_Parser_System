"""LangGraph integration: typed state and a node function.

Use:

    from fermdocs_characterize.langgraph_node import (
        CharacterizationState, characterize_node
    )

    builder = StateGraph(CharacterizationState)
    builder.add_node("characterize", characterize_node)

The state is intentionally minimal. Large artifacts can be passed as `{path: ...}`
pointers if needed; the node here works on the in-memory dict for v1.
"""

from __future__ import annotations

from typing import Any, TypedDict

from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.validators.output_validator import ValidationError


class CharacterizationState(TypedDict, total=False):
    """Minimal LangGraph state for the characterization step.

    `dossier` is the input ingestion dossier dict.
    `output` is the produced CharacterizationOutput.
    `errors` accumulates fatal errors (e.g. validation failures).
    """

    dossier: dict[str, Any]
    output: CharacterizationOutput
    errors: list[str]


def characterize_node(state: CharacterizationState) -> CharacterizationState:
    """LangGraph node: run characterization on `state['dossier']`."""
    pipeline = CharacterizationPipeline()
    try:
        output = pipeline.run(state["dossier"])
    except ValidationError as e:
        return {"errors": list(state.get("errors", [])) + e.errors}
    return {"output": output, "errors": list(state.get("errors", []))}
