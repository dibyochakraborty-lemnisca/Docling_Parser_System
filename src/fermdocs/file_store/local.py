from __future__ import annotations

import os
import shutil
from pathlib import Path

from fermdocs.file_store.base import StoredFile, sha256_of


class LocalFileStore:
    """Content-addressed file store on local disk.

    Files are stored at <root>/files/<sha256>.<ext>. Re-storing the same bytes is a no-op.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None):
        self._root = Path(root or os.environ.get("FERMDOCS_DATA_DIR", "./data")).resolve()
        (self._root / "files").mkdir(parents=True, exist_ok=True)

    def put(self, src: Path) -> StoredFile:
        digest = sha256_of(src)
        ext = src.suffix.lower()
        dest = self._root / "files" / f"{digest}{ext}"
        if not dest.exists():
            shutil.copy2(src, dest)
        return StoredFile(
            sha256=digest, storage_path=str(dest), size_bytes=dest.stat().st_size
        )

    def open(self, storage_path: str) -> bytes:
        with open(storage_path, "rb") as f:
            return f.read()
