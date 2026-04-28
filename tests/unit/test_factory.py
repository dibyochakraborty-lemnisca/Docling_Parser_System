from __future__ import annotations

import pytest

from fermdocs.mapping.factory import UnknownProviderError, build_mapper
from fermdocs.mapping.mapper import FakeHeaderMapper


def test_use_fake_short_circuits():
    mapper = build_mapper(use_fake=True)
    assert isinstance(mapper, FakeHeaderMapper)


def test_unknown_provider_raises():
    with pytest.raises(UnknownProviderError):
        build_mapper(provider="notathing")


def test_fake_provider_explicit():
    mapper = build_mapper(provider="fake")
    assert isinstance(mapper, FakeHeaderMapper)
