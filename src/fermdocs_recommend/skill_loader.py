"""Access to the vendored brewtwin skill docs.

The skills are copied into this package (src/fermdocs_recommend/skills/) so the
stage never depends on the transient ~/Downloads/brewtwin-main checkout. The
README (model-selection hierarchy + shared conventions C1-C6 + gotcha table) is
injected into the agent's system prompt; the per-family recipes are served on
demand via the get_skill tool.
"""

from __future__ import annotations

from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SKILL_NAMES = (
    "fit-mechanistic-model",
    "fit-surrogate-model",
    "fit-hybrid-model",
    "analyze-and-interpret",
)


def load_readme() -> str:
    """The shared conventions + model-selection hierarchy (always in prompt)."""
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
