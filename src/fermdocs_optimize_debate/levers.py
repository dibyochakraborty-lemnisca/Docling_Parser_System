"""The knob → effect-variable spec, shared by topic generation and the seam.

A "lever" is one controllable knob the optimizer can move, plus the measured
variables it acts on (its evidence channels) and a forward-looking question that
reframes the run from "what went wrong" to "where is the titer headroom." The
same spec drives two things:

  - topic generation (topics.py): one opportunity SeedTopic per lever, grounded
    in the characterization evidence for its effect variables;
  - the inform-only seam (schema.py): mapping a debated lever's affected_variables
    back to the knob(s) the optimizer can actually move.

Knob names match `fermdocs_optimize.schema.KNOB_NAMES` so a debated lever lines up
1:1 with the optimizer's search box.
"""
from __future__ import annotations

from dataclasses import dataclass

from fermdocs_optimize.schema import KNOB_NAMES


@dataclass(frozen=True)
class KnobLever:
    knob: str                          # one of KNOB_NAMES
    effect_variables: tuple[str, ...]  # measured channels this knob acts on
    question: str                      # forward-looking frame; "{obj}" = objective species


# Lactic-acid (LABS) levers. effect_variables are the golden-column names the
# characterization trajectories/findings use (X biomass, S substrate, M maltose,
# P product, V volume).
DEFAULT_LEVERS: tuple[KnobLever, ...] = (
    KnobLever("biomass", ("X", "P"),
              "Could a different initial biomass (inoculum density) raise peak {obj}?"),
    KnobLever("total_sub", ("S", "M", "P"),
              "Is substrate left unconsumed — would changing total initial substrate raise peak {obj}?"),
    KnobLever("malt_frac", ("M", "S", "P"),
              "Does the maltose fraction shift flux — would a different glucose/maltose split raise peak {obj}?"),
    KnobLever("dilution", ("V", "S", "P"),
              "Is the feed/dilution rate leaving {obj} on the table (washout vs starvation)?"),
)


def knobs_for_variables(
    variables: list[str],
    *,
    levers: tuple[KnobLever, ...] = DEFAULT_LEVERS,
    objective_species: str = "P",
) -> list[str]:
    """Which knobs act on any of these variables — used to map a debated lever's
    affected_variables back onto the optimizer's knobs. Preserves KNOB_NAMES order.

    The objective species (e.g. P) is excluded from matching: it's the shared
    outcome every lever moves, so leaving it in would map every P-touching
    hypothesis onto all knobs. Knobs are discriminated by their DRIVER variables
    (X, S, M, V), not by the objective."""
    vs = set(variables) - {objective_species}
    hit = {lever.knob for lever in levers
           if vs & (set(lever.effect_variables) - {objective_species})}
    return [k for k in KNOB_NAMES if k in hit]
