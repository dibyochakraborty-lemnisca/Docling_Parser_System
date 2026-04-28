from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fermdocs.domain.models import ParsedTable


class FileParser(ABC):
    """Every parser maps a single file to zero or more ParsedTable objects.

    The locator dict shape varies by source format but always lives in
    ParsedTable.locator. Pipelines never inspect locator structure.
    """

    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path) -> list[ParsedTable]: ...
