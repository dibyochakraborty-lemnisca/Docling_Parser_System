from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fermdocs.domain.models import ParseResult


class FileParser(ABC):
    """Every parser maps a single file to a ParseResult.

    ParseResult carries tables and (optionally) narrative blocks. The locator
    dict shape varies by source format but always lives in ParsedTable.locator
    or NarrativeBlock.locator. Pipelines never inspect locator structure.
    """

    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path) -> ParseResult: ...
