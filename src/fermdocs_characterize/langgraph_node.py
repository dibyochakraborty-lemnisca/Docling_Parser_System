"""LangGraph integration: typed state and a node function.

Use:

    from fermdocs_characterize.langgraph_node import (
        CharacterizationState, characterize_node
    )

    builder = StateGraph(CharacterizationState)
    builder.add_node("characterize", characterize_node)

The state is intentionally minimal. Large artifacts can be passed as `{path: ...}`
pointers if needed; the node here works on the in-memory dict for v1.

The trajectory_analyzer (May 2026) is built from environment configuration
when available — same behavior as fermdocs_characterize.cli.main. Pass
`enable_trajectory_analyzer=False` in the state to force-disable.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from fermdocs_characterize.agents.llm_client import build_characterize_client
from fermdocs_characterize.agents.trajectory_analyzer import (
    build_trajectory_analyzer,
)
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.validators.output_validator import ValidationError

_log = logging.getLogger(__name__)


class CharacterizationState(TypedDict, total=False):
    """Minimal LangGraph state for the characterization step.

    `dossier` is the input ingestion dossier dict.
    `output` is the produced CharacterizationOutput.
    `errors` accumulates fatal errors (e.g. validation failures).
    `enable_trajectory_analyzer`: optional bool; when False, force-disable
        the trajectory_analyzer regardless of env config. Default True.
    """

    dossier: dict[str, Any]
    output: CharacterizationOutput
    errors: list[str]
    enable_trajectory_analyzer: bool


def characterize_node(state: CharacterizationState) -> CharacterizationState:
    """LangGraph node: run characterization on `state['dossier']`."""
    enable_analyzer = state.get("enable_trajectory_analyzer", True)
    analyzer = None
    if enable_analyzer:
        try:
            client = build_characterize_client()
            analyzer = build_trajectory_analyzer(client)
        except Exception as exc:
            _log.warning(
                "trajectory_analyzer disabled (%s); running deterministic only",
                exc.__class__.__name__,
            )
            analyzer = None

    pipeline = CharacterizationPipeline(trajectory_analyzer=analyzer)
    try:
        output = pipeline.run(state["dossier"])
    except ValidationError as e:
        return {"errors": list(state.get("errors", [])) + e.errors}
    return {"output": output, "errors": list(state.get("errors", []))}
