"""F2 — the canonical computation cache.

One idea: every number the system reports must resolve to ONE cached computation,
keyed on its inputs. Re-asking the same question returns the identical value
instead of re-deriving it — which is what killed the +1.532 -> +2.026 -> 2.026
drift (the same effect re-computed narratively with a slightly different recipe
each turn). If two callers ask the same (function, objective, conditioning, data)
question, they resolve to the same cached result or it is a bug, not a "close
enough".

Key design points:
  - The key includes a DATA VERSION fingerprint (content hash of the dossier +
    observations). Re-ingesting or correcting the data (F0) busts the cache, so
    you never serve a stale number after the inputs change.
  - The cache is process-local. Determinism across processes comes from the
    computations being pure functions of their inputs — same inputs -> same key
    -> recompute yields the identical value anyway. The cache adds in-process
    single-sourcing (no drift between callers) and speed.
  - Keys are explicit tuples, not pickles, so they're inspectable and testable.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


def fingerprint_df(df: pd.DataFrame | None) -> str:
    """Stable content hash of a DataFrame (values + column names + shape).

    Two DataFrames with the same contents fingerprint identically, so equivalent
    data resolves to the same cache key regardless of object identity."""
    if df is None:
        return "none"
    try:
        row_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except Exception:  # noqa: BLE001 — unhashable cells -> fall back to a repr hash
        row_hash = repr(df.values.tolist()).encode()
    cols = repr(list(df.columns)).encode()
    return hashlib.sha1(row_hash + cols).hexdigest()  # noqa: S324 — fingerprint, not security


def fingerprint_obj(obj: Any) -> str:
    """Stable content hash of a JSON-able object (e.g. a dossier dict). None-safe."""
    if obj is None:
        return "none"
    try:
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
    except Exception:  # noqa: BLE001
        blob = repr(obj).encode()
    return hashlib.sha1(blob).hexdigest()  # noqa: S324


def data_version(obs_df: pd.DataFrame | None, dossier: Any = None) -> str:
    """The single data-version token threaded into every computation key. Changes
    iff the observations or dossier change, which is exactly when cached numbers
    must be recomputed."""
    return f"{fingerprint_df(obs_df)}:{fingerprint_obj(dossier)}"


def make_key(
    fn_name: str,
    *,
    objective: str | None,
    data_ver: str,
    conditioning: Iterable[str] | None = None,
    extra: tuple = (),
) -> tuple:
    """Canonical, inspectable computation key.

    `conditioning` is the covariate set held constant (A1) — sorted so order is
    irrelevant; empty/None for an unconditioned estimate. `extra` carries any
    additional discriminators (subset id, metric variant)."""
    cond = tuple(sorted(conditioning)) if conditioning else ()
    return (fn_name, objective, cond, data_ver, *extra)


class ComputationCache:
    """Process-local memo of canonical computations, keyed by `make_key`."""

    def __init__(self) -> None:
        self._store: dict[tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: tuple, compute: Callable[[], Any]) -> Any:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = compute()
        self._store[key] = value
        return value

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


# Module-level default cache. Callers route their effect computations through this
# so every consumer of a given (fn, objective, conditioning, data) sees one value.
_CACHE = ComputationCache()


def default_cache() -> ComputationCache:
    return _CACHE


def reset_default_cache() -> None:
    """Tests call this to isolate; production never needs it (keys are content-keyed)."""
    _CACHE.clear()
