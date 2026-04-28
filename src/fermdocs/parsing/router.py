from __future__ import annotations

from pathlib import Path

from fermdocs.domain.models import ParsedTable
from fermdocs.parsing.base import FileParser


class UnsupportedFormatError(ValueError):
    pass


class FormatRouter:
    def __init__(self, parsers: list[FileParser]):
        self._parsers = parsers

    def parse(self, path: Path) -> list[ParsedTable]:
        for parser in self._parsers:
            if parser.supports(path):
                return parser.parse(path)
        raise UnsupportedFormatError(f"No parser registered for {path.suffix}")
