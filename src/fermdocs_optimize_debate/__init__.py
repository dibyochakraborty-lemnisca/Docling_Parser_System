"""Optimization debate — the qualitative front half of the optimizer system.

Reframes the hypothesis debate engine from fault-finding ("what went wrong") to
opportunity-finding ("what levers raise product titer"). It reuses the engine
unchanged (run_stage + hooks + SeedTopic + HypothesisOutput) and supplies only
optimization topics and optimization specialist specs. The debate INFORMS the
closed-loop optimizer; it never constrains the search — the oracle stays the
source of truth.
"""
