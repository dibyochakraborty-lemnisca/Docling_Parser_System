"""fermdocs_characterize: characterization agent for fermentation experiment dossiers.

Reads an ingestion dossier, produces a CharacterizationOutput with findings,
trajectories, timeline, and a structural facts graph. v1 is deterministic-only
(no LLM). See Execution.md for the full plan.
"""

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    Finding,
    FindingType,
    Severity,
    Trajectory,
    TimelineEvent,
    Deviation,
    OpenQuestion,
    DecisionType,
    FactsGraph,
    Node,
    Edge,
    EdgeType,
    NodeType,
    Meta,
    KineticFit,
)

__all__ = [
    "CharacterizationOutput",
    "Finding",
    "FindingType",
    "Severity",
    "Trajectory",
    "TimelineEvent",
    "Deviation",
    "OpenQuestion",
    "DecisionType",
    "FactsGraph",
    "Node",
    "Edge",
    "EdgeType",
    "NodeType",
    "Meta",
    "KineticFit",
]

SCHEMA_VERSION = "1.0"
CHARACTERIZATION_VERSION = "v1.0.0"
