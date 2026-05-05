"""Verified imperative toolkit for the trajectory analyzer.

The catalog (`fermdocs_characterize.agents.metric_catalog`) declares
WHAT can be computed; this package declares HOW. The split exists so
the LLM never invents math: it picks a catalog entry, imports the
named `toolkit_fn`, and reads back the result.

Submodules:
  - kinetics:   Tier A growth-rate / phase-segmentation math (PR 1)
  - operational: Tier A controller / agitation / DO-margin math (PR 2)
  - cross_run:   Tier A cohort comparison math (PR 2)
  - balances:    Tier B yield / RQ / mass balance math (PR 2)
  - literature:  Tier C reference-constant lookups + estimators (PR 3)

Each function is deterministic, small, and unit-tested with synthetic
inputs. No I/O, no LLM calls, no global state. They take numpy arrays
or pandas Series and return either scalars or small dataclasses.
"""
