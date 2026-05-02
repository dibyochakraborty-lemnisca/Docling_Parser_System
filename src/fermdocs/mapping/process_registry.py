"""Process registry: closed list of processes the identity extractor can emit.

The registry is hand-curated. The LLM picks among these or returns null;
it cannot invent new entries. Adding a process is a YAML edit + a code
review, not a runtime decision.

A registry entry carries:
  - id: stable string used as ProcessIdentity.process_id
  - identity fields (organism, product, process_family, typical_scale_l)
  - aliases: alternate phrasings the LLM may encounter in source text
  - variable_fingerprint: required/strong/forbidden golden-column names
    that act as a sanity check on the LLM's pick

The fingerprint is the post-LLM safety net for plausible-but-wrong picks.
If the LLM says "this is penicillin" but the dossier has no PAA observations
at all, the fingerprint check downgrades the identity to UNKNOWN.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from fermdocs.domain.golden_schema import cached_schema

_DEFAULT_PATH = Path(__file__).parent.parent / "schema" / "process_registry.yaml"


class VariableFingerprint(BaseModel):
    required: list[str] = Field(default_factory=list)
    strong: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class ProcessAliases(BaseModel):
    organism: list[str] = Field(default_factory=list)
    product: list[str] = Field(default_factory=list)


class ProcessRegistryEntry(BaseModel):
    id: str
    organism: str
    product: str
    process_family: str
    typical_scale_l: float | None = None
    aliases: ProcessAliases = Field(default_factory=ProcessAliases)
    variable_fingerprint: VariableFingerprint


class ProcessRegistry(BaseModel):
    version: str
    processes: list[ProcessRegistryEntry]

    @field_validator("processes")
    @classmethod
    def _unique_ids(
        cls, v: list[ProcessRegistryEntry]
    ) -> list[ProcessRegistryEntry]:
        seen: set[str] = set()
        for p in v:
            if p.id in seen:
                raise ValueError(f"duplicate process id: {p.id!r}")
            seen.add(p.id)
        return v

    @model_validator(mode="after")
    def _no_alias_collisions(self) -> ProcessRegistry:
        # An alias claimed by two processes is ambiguous; reject at load time.
        for kind in ("organism", "product"):
            owners: dict[str, str] = {}
            for p in self.processes:
                aliases = getattr(p.aliases, kind)
                for alias in aliases:
                    key = alias.strip().lower()
                    if not key:
                        continue
                    if key in owners and owners[key] != p.id:
                        raise ValueError(
                            f"{kind} alias {alias!r} claimed by both"
                            f" {owners[key]!r} and {p.id!r}"
                        )
                    owners[key] = p.id
        return self

    def by_id(self) -> dict[str, ProcessRegistryEntry]:
        return {p.id: p for p in self.processes}


def resolve_registry_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("FERMDOCS_PROCESS_REGISTRY_PATH")
    if env:
        return Path(env)
    return _DEFAULT_PATH


def load_registry(
    path: str | os.PathLike[str] | None = None,
    *,
    validate_against_schema: bool = True,
) -> ProcessRegistry:
    """Load and validate a process registry.

    By default checks that every fingerprint variable exists in the loaded
    golden schema. Pass `validate_against_schema=False` only in tests that
    want to construct registry entries naming variables outside the live
    schema.
    """
    p = resolve_registry_path(path)
    with open(p) as f:
        data = yaml.safe_load(f)
    registry = ProcessRegistry.model_validate(data)

    if validate_against_schema:
        schema_vars = set(cached_schema().by_name())
        for entry in registry.processes:
            for kind in ("required", "strong", "forbidden"):
                for var in getattr(entry.variable_fingerprint, kind):
                    if var not in schema_vars:
                        raise ValueError(
                            f"process {entry.id!r}: fingerprint {kind} variable"
                            f" {var!r} not in golden schema"
                        )

    return registry


@lru_cache(maxsize=4)
def cached_registry(path: str | None = None) -> ProcessRegistry:
    return load_registry(path)


def fingerprint_check(
    entry: ProcessRegistryEntry, present_variables: set[str]
) -> tuple[bool, str | None]:
    """Returns (ok, reason_if_rejected).

    Rejects when:
      - any `required` variable is absent from the dossier
      - any `forbidden` variable is present in the dossier
    `strong` variables are advisory; their absence does NOT reject.
    """
    fp = entry.variable_fingerprint
    missing = [v for v in fp.required if v not in present_variables]
    if missing:
        return False, f"missing required vars: {missing}"
    forbidden_present = [v for v in fp.forbidden if v in present_variables]
    if forbidden_present:
        return False, f"forbidden vars present: {forbidden_present}"
    return True, None


def parse_registry_dict(data: dict[str, Any]) -> ProcessRegistry:
    """Test helper: build a ProcessRegistry from an in-memory dict.

    Skips the golden-schema variable check so test fixtures can name
    variables that aren't in production YAML.
    """
    return ProcessRegistry.model_validate(data)
