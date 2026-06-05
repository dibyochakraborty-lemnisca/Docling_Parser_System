"""Access to the vendored optimizer skill docs.

The README (shared conventions + the loop + the integrity invariant) is injected
into the agent's system prompt; the per-task recipes (optimize-titer,
choose-model-and-proposer) are served on demand via the get_skill tool.
"""

from __future__ import annotations

from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SKILL_NAMES = (
    "optimize-titer",
    "choose-model-and-proposer",
)


def load_readme() -> str:
    """Shared conventions + the loop + integrity invariant (always in prompt)."""
    p = _SKILLS_DIR / "README.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_skill(name: str) -> str | None:
    """Full text of one skill recipe, or None if the name is unknown."""
    name = name.strip().removesuffix(".md")
    p = _SKILLS_DIR / f"{name}.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def available_skills() -> list[str]:
    return list(SKILL_NAMES)
