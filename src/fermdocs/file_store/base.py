from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class StoredFile:
    sha256: str
    storage_path: str
    size_bytes: int


class FileStore(Protocol):
    """Abstract storage so v2 can swap LocalFileStore -> S3FileStore."""

    def put(self, src: Path) -> StoredFile: ...

    def open(self, storage_path: str) -> bytes: ...


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
