"""OptimizeHooks — the engine's RunnerHooks, with optimization specialists.

`LiveHooks` already consumes exactly the attributes `OptimizeLoadedBundle`
exposes (hyp_input, the *_pool lists, characterization), so we subclass it and
swap ONLY the three specialist agents for their optimization-framed variants.
The orchestrator, synthesizer, critic, judge, projectors, tool bundle, and
citation lookups are reused unchanged — the diagnostic stage is untouched.
"""
from __future__ import annotations

from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.schema import SpecialistRole

from fermdocs_optimize_debate.specs import (
    build_opt_kinetics_specialist,
    build_opt_mass_transfer_specialist,
    build_opt_metabolic_specialist,
)


class OptimizeHooks(LiveHooks):
    def __init__(self, bundle, *, client=None, memory=None, run_id=None):
        super().__init__(bundle, client=client, memory=memory, run_id=run_id)
        # Swap the three specialists for optimization personas; reuse everything
        # else the base constructed.
        self._kinetics = build_opt_kinetics_specialist(self._client, self._tools)
        self._mass_transfer = build_opt_mass_transfer_specialist(self._client, self._tools)
        self._metabolic = build_opt_metabolic_specialist(self._client, self._tools)
        self._specialists: dict[SpecialistRole, object] = {
            "kinetics": self._kinetics,
            "mass_transfer": self._mass_transfer,
            "metabolic": self._metabolic,
        }
