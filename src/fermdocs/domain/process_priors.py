"""Process priors registry — organism × process_family → variable bounds.

The diagnosis agent's third grounding path (alongside schema_only and
cross_run). When a single-run dossier hits the agent and there's no cohort
to compare against, priors give it organism-aware reference values: "yeast
biomass typically reaches 80-180 g/L per Verduyn 1991; observed 60 g/L is
below the floor."

Loader pattern mirrors `fermdocs.domain.golden_schema`:
  - load_priors(path?) → ProcessPriors
  - cached_priors(path?) → ProcessPriors  (lru_cache)
  - resolve_priors(priors, organism, process_family?, variable?) → list[ResolvedPrior]

Alias matching is case-insensitive substring on the organism field. The
resolver is conservative — empty list on miss, never raises. Callers
(the agent's get_priors tool) decide what to do with no-match.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_PATH = Path(__file__).parent.parent / "schema" / "process_priors.yaml"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class PriorBound(BaseModel):
    """Bounds for a single variable.

    `range: [low, high]` is a SOFT bound — values outside trigger
    candidate findings, not hard failures. `typical` is the point
    estimate for residual / sigma calculations. `source` is required;
    no prior ships without a citation.
    """

    model_config = ConfigDict(frozen=True)

    range: tuple[float, float] = Field(
        description="Soft [low, high] bound. Use `typical` for point comparisons."
    )
    typical: float
    source: str = Field(min_length=1, description="Required citation.")
    note: str | None = None

    def model_post_init(self, _ctx: object) -> None:
        # tuple coerces to ordered, but be defensive: low <= high
        low, high = self.range
        if low > high:
            raise ValueError(f"prior range low={low} > high={high}")


class ProcessFamily(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    priors: dict[str, PriorBound] = Field(default_factory=dict)


class OrganismPriors(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    process_families: list[ProcessFamily] = Field(default_factory=list)


class ProcessPriors(BaseModel):
    """Top-level priors document. Round-trip target for process_priors.yaml."""

    model_config = ConfigDict(frozen=True)

    version: Literal["1.0"]
    organisms: list[OrganismPriors] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Resolved view used by the agent
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPrior:
    """Flat per-variable record returned by `resolve_priors`.

    The agent's `get_priors` tool serializes a list of these. The flat
    shape (vs nested by organism / family) makes downstream filtering
    cheap and JSON-safe.
    """

    organism: str
    process_family: str
    variable: str
    range: tuple[float, float]
    typical: float
    source: str
    note: str | None

    def to_dict(self) -> dict:
        return {
            "organism": self.organism,
            "process_family": self.process_family,
            "variable": self.variable,
            "range": list(self.range),
            "typical": self.typical,
            "source": self.source,
            "note": self.note,
        }


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------


def resolve_priors_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("FERMDOCS_PRIORS_PATH")
    if env:
        return Path(env)
    return _DEFAULT_PATH


def load_priors(path: str | os.PathLike[str] | None = None) -> ProcessPriors:
    priors_path = resolve_priors_path(path)
    with open(priors_path) as f:
        data = yaml.safe_load(f)
    return ProcessPriors.model_validate(data)


@lru_cache(maxsize=4)
def cached_priors(path: str | None = None) -> ProcessPriors:
    return load_priors(path)


def priors_version(path: str | os.PathLike[str] | None = None) -> str:
    """Light-weight version reader. Used by BundleMeta to record provenance.

    Parses YAML but does not validate the full ProcessPriors model —
    cheaper than load_priors when the caller only needs the version string.
    """
    priors_path = resolve_priors_path(path)
    with open(priors_path) as f:
        data = yaml.safe_load(f)
    v = data.get("version")
    if not isinstance(v, str):
        raise ValueError(f"process_priors {priors_path} missing string `version`")
    return v


# -----------------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------------


def _organism_matches(organism_query: str, organism: OrganismPriors) -> bool:
    """Case-insensitive substring match against name + aliases.

    Substring is the right shape because dossier organism fields are messy
    ("Saccharomyces cerevisiae strain X33", "E. coli BL21(DE3)"). Exact
    match would miss too many real cases.
    """
    q = organism_query.lower().strip()
    if not q:
        return False
    if q in organism.name.lower():
        return True
    if organism.name.lower() in q:
        return True
    for alias in organism.aliases:
        a = alias.lower()
        if a in q or q in a:
            return True
    return False


def resolve_priors(
    priors: ProcessPriors,
    *,
    organism: str | None = None,
    process_family: str | None = None,
    variable: str | None = None,
) -> list[ResolvedPrior]:
    """Flat lookup. Returns one row per (organism, process_family, variable).

    Filters:
      organism: case-insensitive substring on organism.name + aliases
      process_family: exact match on family.name (case-insensitive)
      variable: exact match on variable name (case-insensitive)

    Empty list on no match. Never raises.
    """
    out: list[ResolvedPrior] = []
    for org in priors.organisms:
        if organism is not None and not _organism_matches(organism, org):
            continue
        for fam in org.process_families:
            if (
                process_family is not None
                and process_family.lower() != fam.name.lower()
            ):
                continue
            for var_name, bound in fam.priors.items():
                if (
                    variable is not None
                    and variable.lower() != var_name.lower()
                ):
                    continue
                out.append(
                    ResolvedPrior(
                        organism=org.name,
                        process_family=fam.name,
                        variable=var_name,
                        range=bound.range,
                        typical=bound.typical,
                        source=bound.source,
                        note=bound.note,
                    )
                )
    return out


__all__ = [
    "OrganismPriors",
    "PriorBound",
    "ProcessFamily",
    "ProcessPriors",
    "ResolvedPrior",
    "cached_priors",
    "load_priors",
    "priors_version",
    "resolve_priors",
    "resolve_priors_path",
]
