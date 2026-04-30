"""FactsGraph builder. v1 returns an empty graph; the structural map is
populated in v2+ when downstream agents have a clear use for it.

Kept as a separate module so v2/v3 can populate without touching pipeline.py.
"""

from __future__ import annotations

from fermdocs_characterize.schema import FactsGraph
from fermdocs_characterize.views.summary import Summary


def build_facts_graph(_summary: Summary) -> FactsGraph:
    return FactsGraph(nodes=[], edges=[])
