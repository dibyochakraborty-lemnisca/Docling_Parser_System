from __future__ import annotations

import os

from fermdocs.mapping.mapper import FakeHeaderMapper, HeaderMapper


class UnknownProviderError(ValueError):
    pass


def build_mapper(provider: str | None = None, *, use_fake: bool = False) -> HeaderMapper:
    """Pick a mapper implementation by provider name.

    Resolution order: explicit arg > FERMDOCS_MAPPER_PROVIDER env var > 'anthropic'.
    Use --fake-mapper / use_fake=True to short-circuit to FakeHeaderMapper.
    """
    if use_fake:
        return FakeHeaderMapper()
    name = (provider or os.environ.get("FERMDOCS_MAPPER_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        from fermdocs.mapping.client import LLMHeaderMapper

        return LLMHeaderMapper()
    if name == "gemini":
        from fermdocs.mapping.gemini_client import GeminiHeaderMapper

        return GeminiHeaderMapper()
    if name == "fake":
        return FakeHeaderMapper()
    raise UnknownProviderError(
        f"unknown mapper provider: {name!r} (expected 'anthropic', 'gemini', or 'fake')"
    )
