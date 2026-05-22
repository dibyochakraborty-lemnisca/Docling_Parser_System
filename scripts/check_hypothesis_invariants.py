"""CI guard for hypothesis-stage architectural invariants.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 4.

Invariants checked:

1. NO agent reads another agent's internal state directly.
   Agents talk to each other ONLY via:
     - Projector views (typed objects from projector.py)
     - RunnerHooks (the protocol the runner calls into)
     - global.md events (read-only via state.py / projector)

   Concretely: a file under agents/ may NOT import another file under
   agents/ (other than the shared specialist_base).

2. NO live-LLM module imports stub agents.
   stubs/canned_agents.py exists for tests only. Production agent
   modules and live_hooks must not import from it.

3. agents/orchestrator.py must NOT import from agents/specialist_*
   (the orchestrator only routes via topic_id, never inspects facets).

This guard is intentionally a tiny grep, not a static analyzer. It's
enough to catch the kinds of cross-agent leakage that's quiet and
corrosive. Stage 4 / v1 may add a real AST-based check.

Exit codes:
  0 — all invariants hold
  1 — violations found (printed to stderr)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "src" / "fermdocs_hypothesis" / "agents"
LIVE_HOOKS = ROOT / "src" / "fermdocs_hypothesis" / "live_hooks.py"
STUB_PATH_FRAGMENT = "fermdocs_hypothesis.stubs"

# Files agents are allowed to import. Everything else under agents/ is
# off-limits to other agents (only specialist_base is the shared base class).
AGENTS_ALLOWED_INTERNAL = {"specialist_base", "__init__"}

ORCHESTRATOR_FORBIDDEN = re.compile(
    r"^\s*from\s+fermdocs_hypothesis\.agents\.specialist_[a-z_]+\s+import",
    re.MULTILINE,
)


def _check_no_cross_agent_imports() -> list[str]:
    violations: list[str] = []
    agent_files = sorted(p for p in AGENTS_DIR.glob("*.py") if p.is_file())
    for agent_file in agent_files:
        text = agent_file.read_text()
        # Look for imports of OTHER files under agents/
        for match in re.finditer(
            r"^\s*from\s+fermdocs_hypothesis\.agents\.([a-z_]+)\s+import",
            text,
            re.MULTILINE,
        ):
            target = match.group(1)
            if target in AGENTS_ALLOWED_INTERNAL:
                continue
            # Self-import is fine
            if target == agent_file.stem:
                continue
            violations.append(
                f"{agent_file.relative_to(ROOT)}: imports from agents.{target}"
                " — agents must talk via Projector views / RunnerHooks, not"
                " each other directly"
            )
    return violations


def _check_no_stub_imports_in_production() -> list[str]:
    violations: list[str] = []
    targets = list(AGENTS_DIR.glob("*.py")) + [LIVE_HOOKS]
    for f in targets:
        if not f.exists() or f.name == "__init__.py":
            continue
        text = f.read_text()
        if STUB_PATH_FRAGMENT in text:
            violations.append(
                f"{f.relative_to(ROOT)}: imports from {STUB_PATH_FRAGMENT}"
                " — stubs are test-only"
            )
    return violations


def _check_orchestrator_doesnt_inspect_specialists() -> list[str]:
    f = AGENTS_DIR / "orchestrator.py"
    if not f.exists():
        return []
    text = f.read_text()
    matches = ORCHESTRATOR_FORBIDDEN.findall(text)
    if matches:
        return [
            f"agents/orchestrator.py: imports a specialist module"
            f" ({matches[0]!r}) — orchestrator must route via topic_id only"
        ]
    return []


def main() -> int:
    all_violations: list[str] = []
    all_violations += _check_no_cross_agent_imports()
    all_violations += _check_no_stub_imports_in_production()
    all_violations += _check_orchestrator_doesnt_inspect_specialists()

    if all_violations:
        print(
            "✗ hypothesis-stage architectural invariants violated:",
            file=sys.stderr,
        )
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("✓ hypothesis-stage architectural invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
