"""Recommendation stage (Stage 5).

Fits brewtwin mechanistic / surrogate / hybrid models to a run's trajectories,
picks the best-supported family via a deterministic rubric (or honestly refuses),
and proposes simulation-backed interventions. Runs after the hypothesis stage on
a DONE run.
"""

from fermdocs_recommend.agent import RecommendationAgent
from fermdocs_recommend.schema import RecommendationOutput

__all__ = ["RecommendationAgent", "RecommendationOutput"]
