# Edge Vocabulary (closed)

This is the closed list of `EdgeType` values for the structural `facts_graph`. The graph is **structural only** — it represents what was measured under what condition at what time. Findings (insights) are NOT graph edges; they live in the flat `findings` list.

Validators reject any edge whose type is not on this list.

| EdgeType | Source node → target | Meaning |
|---|---|---|
| `measured` | `Sample → Measurement` | This sample produced this measurement. The Measurement node carries the variable name, value, unit, and ingestion observation_id in its attributes. |
| `under_condition` | `Sample → Condition` | This sample was taken under a stated experimental condition (strain, media, scale, target setpoint). One sample can have multiple conditions. |
| `at_time` | `Measurement → (timestamp)` | Encoded as an attribute on the Measurement node OR as an edge to a synthetic time node. v1 uses the attribute form; the edge form is reserved if a future query layer wants temporal nodes as first-class. |
| `derived_from` | `Sample → Source` | This sample's data came from this source file (PDF, Excel, CSV). Mirrors ingestion's file_id. |

## Why no relationship edges

Relationships like `covaries_with`, `outperforms_at_condition`, `contradicts`, `precedes` are **findings**, not graph edges. They carry confidence, evidence_observation_ids, caveats, and competing_explanations — too rich to live as edge attributes, and queryable as a flat list with rich metadata.

Graph edges are deterministic, low-information structural connectors. Findings are the agent's actual outputs.

## Adding a new edge type

Requires:
1. A code change in `schema.py` adding the enum value.
2. A clear semantic for which node types it can connect.
3. The `_no_dangling_edges` and any other validators updated.
4. A reason that the relationship can't be expressed as a Finding instead.

If the new edge would carry confidence, evidence, or competing explanations, it's a Finding, not an edge.
