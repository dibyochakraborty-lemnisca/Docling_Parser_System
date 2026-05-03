"""Observer — deterministic writer for global.md.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §3.

global.md is markdown for human reading with embedded JSONL fenced blocks for
machine parsing. Single writer. Atomic temp+rename per write so crashed runs
never leave a half-written log. Append-only invariant: existing events are
never mutated.

Format:
    # Hypothesis stage event log

    <one fenced ```jsonl block per write, each line one event JSON>

The reader concatenates every fenced jsonl block in order. Multiple blocks
are fine; the parser doesn't care where the boundaries fall.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from fermdocs_hypothesis.events import Event, EventEnvelope

_HEADER = "# Hypothesis stage event log\n\n"
_FENCE_OPEN = "```jsonl\n"
_FENCE_CLOSE = "```\n"

_JSONL_BLOCK_RE = re.compile(r"```jsonl\n(.*?)```", re.DOTALL)

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


class Observer:
    """Append-only writer for the event log.

    Use:
        obs = Observer(path)
        obs.write(event)            # one event
        obs.write_many([e1, e2])    # batch

    Every write is atomic (temp+rename). The whole file is rewritten each call;
    that's fine for v0 because event volume is small (≤ ~200 events per stage).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = []
        if self.path.exists():
            self._events = list(read_events(self.path))

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def write(self, event: Event) -> None:
        self._events.append(event)
        self._flush()

    def write_many(self, events: Sequence[Event]) -> None:
        self._events.extend(events)
        self._flush()

    def _flush(self) -> None:
        body = _render(self._events)
        # Atomic temp+rename in same dir to keep filesystem semantics tight.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".global.md.", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _render(events: Sequence[Event]) -> str:
    if not events:
        return _HEADER
    lines = [
        json.dumps(
            event.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=False,
        )
        for event in events
    ]
    return _HEADER + _FENCE_OPEN + "\n".join(lines) + "\n" + _FENCE_CLOSE


def read_events(path: Path) -> list[Event]:
    """Parse all events from a global.md file. Concatenates every jsonl
    fenced block in document order.
    """
    text = Path(path).read_text(encoding="utf-8")
    events: list[Event] = []
    for match in _JSONL_BLOCK_RE.finditer(text):
        block = match.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            events.append(_event_adapter.validate_python(obj))
    return events
