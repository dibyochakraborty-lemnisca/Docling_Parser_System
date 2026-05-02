"""Audit-invariant CI guard.

Bundle directory layout (plans/2026-05-02-execute-python-default.md §2):
    out/bundle_<...>/
      audit/                # write-only at runtime — must NOT be read by any
                            # node that influences the agent's reasoning

This script scans runtime modules for reads from `audit/` paths. It is a
deliberately blunt grep: false positives are fixable by whitelisting; false
negatives would let context poisoning slip in undetected.

Allow-list:
    - test files (anywhere under tests/)
    - the writer modules themselves (they own audit/)
    - this guard script

Anything else that mentions an audit/ read pattern fails the build.

Exit 0 if clean, 1 if violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = [
    REPO_ROOT / "src" / "fermdocs" / "bundle",
    REPO_ROOT / "src" / "fermdocs_diagnose",
    REPO_ROOT / "src" / "fermdocs_characterize",
]

# Allow-listed paths (relative to repo root) — these may legitimately mention
# audit/ paths because they own the writes or document the invariant.
ALLOW_LIST_PREFIXES: tuple[str, ...] = (
    "src/fermdocs/bundle/writer.py",
    "src/fermdocs/bundle/reader.py",  # may expose audit/ paths via @property; reads forbidden
    "src/fermdocs/bundle/__init__.py",
    "src/fermdocs/bundle/meta.py",
    # Stage 2/3 will add: src/fermdocs_diagnose/audit/*.py (TraceWriter)
)

# Patterns that indicate a READ from audit/. We look for filesystem-style
# accesses: open("...audit/..."), Path(".../audit/..."), read_text on audit/ paths,
# json.loads on audit/ files, etc. Pure mentions in strings/docstrings without
# a read context still fail — flag them; whitelist by module path if intentional.
READ_PATTERNS = [
    re.compile(r"""open\s*\(\s*[^)]*['"][^'"]*audit/""", re.IGNORECASE),
    re.compile(r"""Path\s*\([^)]*['"][^'"]*audit/""", re.IGNORECASE),
    re.compile(r"""\.read_text\s*\(\s*\)\s*$"""),  # pair with audit/ in line context
    re.compile(r"""['"][^'"]*audit/[^'"]*['"]"""),  # any string referencing an audit/ path
]


def _is_allowed(path: Path) -> bool:
    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    if "/tests/" in f"/{rel}" or rel.startswith("tests/"):
        return True
    if rel == "scripts/check_audit_invariant.py":
        return True
    return any(rel.startswith(p) for p in ALLOW_LIST_PREFIXES)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        # Skip comment-only lines (the invariant prose is allowed everywhere)
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for pat in READ_PATTERNS:
            if "audit/" in line and pat.search(line):
                violations.append((lineno, line.rstrip()))
                break
    return violations


def main() -> int:
    found: list[tuple[Path, int, str]] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            if _is_allowed(py):
                continue
            for lineno, line in _scan_file(py):
                found.append((py, lineno, line))

    if not found:
        print("audit-invariant guard: clean (no audit/ reads found in runtime modules)")
        return 0

    print("audit-invariant violations:")
    for path, lineno, line in found:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        print(f"  {rel}:{lineno}: {line}")
    print(
        "\nIf this read is legitimate (e.g. a new writer module), add its path"
        " to ALLOW_LIST_PREFIXES in scripts/check_audit_invariant.py.\n"
        "Otherwise, audit/ MUST NOT be read at runtime — see"
        " plans/2026-05-02-execute-python-default.md §2."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
