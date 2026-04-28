from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from fermdocs.domain.models import GoldenSchema

_DEFAULT_PATH = Path(__file__).parent.parent / "schema" / "golden_schema.yaml"


def resolve_schema_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("FERMDOCS_SCHEMA_PATH")
    if env:
        return Path(env)
    return _DEFAULT_PATH


def load_schema(path: str | os.PathLike[str] | None = None) -> GoldenSchema:
    schema_path = resolve_schema_path(path)
    with open(schema_path) as f:
        data = yaml.safe_load(f)
    return GoldenSchema.model_validate(data)


@lru_cache(maxsize=4)
def cached_schema(path: str | None = None) -> GoldenSchema:
    return load_schema(path)
