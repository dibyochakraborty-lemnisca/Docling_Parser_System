"""Characterize-stage LLM agents.

Until May 2026 the characterize stage was purely deterministic. The
`trajectory_analyzer` is the first crossing of the LLM line on this side
of the pipeline — see Execution.md for the architectural shift.

The deterministic stages still run first (spec checks, range_violation
finding generation). The analyzer runs AFTER, with `execute_python`
access, to surface trajectory-grounded patterns the schema layer can't
see (cross-batch variance, phase boundaries, outlier batches,
correlations).
"""
