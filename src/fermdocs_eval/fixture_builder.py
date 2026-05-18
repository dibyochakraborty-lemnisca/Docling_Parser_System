"""Build E2 fixture bundles from typed DefectSpecs.

Strategy: clone-and-mutate. Take a base template bundle (the real indpensim
ingest), apply a small typed mutation per fixture to plant a defect on one
critic axis (or "clean"), write the mutated bundle to disk under
eval/fixtures/e2/<fixture_id>/.

The mutation only touches the smallest surface area needed to plant the
defect. Everything else stays identical to the template — so the fixture
"feels like" a real bundle to upstream agents, and the pipeline takes the
same code paths it would in production.

Schema safety: after writing, every fixture is re-loaded through the real
bundle_loader. If load_bundle raises, the build fails fast.

Force-commit strategy (per plan): each defect spec also carries a
`leading_question` field. The runner passes this to the hypothesis stage
via --question so the synthesizer is steered toward the defective claim,
reducing the "synthesizer fixes it before critic sees it" confound.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fermdocs_hypothesis.bundle_loader import load_bundle

# Mirror src/fermdocs_hypothesis/agents/critic.py axis names.
AxisLabel = Literal[
    "clean",
    "trajectory-axis",
    "robustness-axis",
    "tool-gap-axis",
    "memory-axis",
    "metadata-axis",
    "actionability-axis",
    "question-axis",
]


@dataclass(frozen=True)
class DefectSpec:
    """One fixture's recipe.

    fixture_id: stable slug like "e2-trajectory-clear-01"
    labeled_axis: ground-truth axis the critic SHOULD fire on (or "clean")
    difficulty: "clean" | "clear" | "borderline"
    leading_question: user question passed to hypothesis stage. Designed
        to steer the synthesizer toward the planted defect so the critic
        gets a chance to fire.
    mutation_kind: which mutation function to apply. None for clean fixtures.
    mutation_params: kwargs for the mutation function.
    memory_seed: optional list of (process_family, lesson_text) pairs that
        the runner pre-populates into a StubBackend before invoking the
        pipeline. Used for memory-axis fixtures where the planted defect
        is a misapplied prior. process_family MISMATCH is the defect:
        e.g., seed a "yeast" lesson and run on an "indpensim" bundle.
    notes: free-form authoring notes; carried into the fixture's
        defect_spec.json for traceability.
    """

    fixture_id: str
    labeled_axis: AxisLabel
    difficulty: Literal["clean", "clear", "borderline"]
    leading_question: str
    mutation_kind: str | None = None
    mutation_params: dict = field(default_factory=dict)
    memory_seed: tuple = ()  # ((process_family, lesson_text), ...)
    notes: str = ""


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)


def _clone_template(template_dir: Path, out_dir: Path) -> None:
    """Copy template bundle to out_dir. Overwrites if out_dir exists."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)


def build_fixture(
    spec: DefectSpec,
    template_dir: Path,
    out_root: Path,
) -> Path:
    """Build one fixture bundle. Returns the bundle directory path.

    Raises if the mutation produces a bundle that fails schema validation.
    """
    out_dir = out_root / spec.fixture_id
    _clone_template(template_dir, out_dir)

    if spec.mutation_kind is not None:
        mutator = _MUTATORS[spec.mutation_kind]
        mutator(out_dir, **spec.mutation_params)

    _write_json(
        out_dir / "defect_spec.json",
        {
            "fixture_id": spec.fixture_id,
            "labeled_axis": spec.labeled_axis,
            "difficulty": spec.difficulty,
            "leading_question": spec.leading_question,
            "mutation_kind": spec.mutation_kind,
            "mutation_params": spec.mutation_params,
            "memory_seed": [list(s) for s in spec.memory_seed],
            "notes": spec.notes,
        },
    )

    # Schema-validate by round-tripping through the real loader.
    load_bundle(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Mutators — one per axis kind. Each takes the cloned bundle dir and the
# spec's mutation_params. Keep them small and obvious. Add new mutators by
# registering in _MUTATORS at the bottom.
# ---------------------------------------------------------------------------


def _mut_strip_findings(bundle_dir: Path, *, keep: int = 2) -> None:
    """Robustness-axis: shrink findings pool so claims rest on thin evidence.

    Keeps the first `keep` findings, drops the rest. Also strips downstream
    references (timeline events, diagnosis claims, cross-finding edges) that
    pointed at dropped findings so the bundle stays internally consistent and
    passes cross-output validation.
    """
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)
    diag_path = bundle_dir / "diagnosis" / "diagnosis.json"
    diag = _read_json(diag_path)

    findings = char.get("findings", [])
    if not isinstance(findings, list) or not findings:
        return

    kept = findings[:keep]
    kept_ids = {f["finding_id"] for f in kept if isinstance(f, dict) and "finding_id" in f}
    char["findings"] = kept

    # Drop timeline events whose finding_id is not in the kept set.
    if "timeline" in char and isinstance(char["timeline"], list):
        char["timeline"] = [
            ev
            for ev in char["timeline"]
            if ev.get("finding_id", "") in kept_ids
        ]

    # facts_graph nodes/edges may reference findings. Drop nodes whose id
    # matches a dropped finding_id, and drop edges that touch them.
    fg = char.get("facts_graph") or {}
    if isinstance(fg, dict):
        nodes = fg.get("nodes") or []
        edges = fg.get("edges") or []
        # Some nodes are findings (node_id == finding_id); others are samples,
        # measurements, etc. Only drop finding-typed nodes that we removed.
        finding_node_ids_to_drop = set()
        for n in nodes:
            nid = n.get("node_id") or n.get("id")
            if nid and ":" in nid and nid.split(":", 1)[1].startswith("F-") and nid not in kept_ids:
                finding_node_ids_to_drop.add(nid)
        fg["nodes"] = [
            n
            for n in nodes
            if (n.get("node_id") or n.get("id")) not in finding_node_ids_to_drop
        ]
        fg["edges"] = [
            e
            for e in edges
            if e.get("source") not in finding_node_ids_to_drop
            and e.get("target") not in finding_node_ids_to_drop
        ]
        char["facts_graph"] = fg

    # expected_vs_observed entries may reference findings; drop dangling refs.
    if "expected_vs_observed" in char and isinstance(char["expected_vs_observed"], list):
        char["expected_vs_observed"] = [
            ev
            for ev in char["expected_vs_observed"]
            if ev.get("finding_id", "") in kept_ids
            or "finding_id" not in ev
        ]

    _write_json(char_path, char)

    # Drop diagnosis claims that cite dropped findings.
    if "claims" in diag and isinstance(diag["claims"], list):
        diag["claims"] = [
            c
            for c in diag["claims"]
            if _claim_cites_only_kept(c, kept_ids)
        ]
    _write_json(diag_path, diag)


def _claim_cites_only_kept(claim: dict, kept_ids: set[str]) -> bool:
    """A diagnosis claim survives if every finding it cites is in kept_ids.

    Diagnosis claims cite findings via 'finding_ids' (full
    '<schema_uuid>:F-NNNN' form, matching Finding.finding_id). If a claim has
    no cited findings at all, drop it too — unevidenced claims would also be
    rejected downstream.
    """
    cited = claim.get("finding_ids") or []
    if not cited:
        return False
    return all(c in kept_ids for c in cited)


def _mut_drop_metadata_anomalies(bundle_dir: Path) -> None:
    """Metadata-axis prep — does NOT plant the defect, only ensures the
    template has *some* metadata anomaly for the planted hypothesis to ignore.
    For now this is a no-op stub; the actual defect for metadata-axis is
    expressed via leading_question that ignores existing anomalies.
    """
    return None


def _mut_strip_trajectories(bundle_dir: Path, *, keep: int = 1) -> None:
    """Trajectory-axis: trim trajectory views so the synthesizer cannot
    ground a trajectory claim — but the leading_question demands one.
    Critic should fire trajectory-axis when claims outrun trajectory evidence.
    """
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)
    if "trajectories" in char and isinstance(char["trajectories"], list):
        char["trajectories"] = char["trajectories"][:keep]
    _write_json(char_path, char)


def _mut_drop_narratives(bundle_dir: Path) -> None:
    """Tool-gap-axis prep: removing narratives forces the synthesizer to
    lean on numerical claims, which it then can only ground via execute_python.
    If it doesn't call the tool, critic should fire tool-gap-axis.
    """
    narr_path = bundle_dir / "characterization" / "narrative_observations.json"
    if narr_path.exists():
        narr_path.unlink()
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)
    char["narrative_observations"] = []
    _write_json(char_path, char)


def _mut_noop(bundle_dir: Path) -> None:
    """Clean fixtures and axes driven entirely by leading_question use this
    no-op so the template is unchanged. The defect lives in the question,
    not the bundle."""
    return None


def _mut_plant_weak_n(bundle_dir: Path, *, n: int = 4, target_count: int = 2) -> None:
    """Robustness-axis: plant `weak_n_flag` on a few existing findings.

    Picks the first `target_count` findings whose statistics dict is non-empty
    and overlays `weak_n_flag=True` and `n=<n>`. Critic should fire
    [robustness-axis] when the synthesizer cites these without caveats.
    """
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)
    planted = 0
    for f in char.get("findings", []):
        if planted >= target_count:
            break
        stats = f.get("statistics")
        if stats is None:
            f["statistics"] = {"weak_n_flag": True, "n": n}
        elif isinstance(stats, dict):
            stats["weak_n_flag"] = True
            stats["n"] = n
        planted += 1
    _write_json(char_path, char)


def _mut_plant_symmetry_violation(bundle_dir: Path, *, target_count: int = 2) -> None:
    """Tool-gap-axis: plant `symmetry_violation` on a few findings.

    Critic should fire [tool-gap-axis] if the synthesizer punts to
    `question_answered: insufficient_data` citing these findings.
    """
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)
    planted = 0
    for f in char.get("findings", []):
        if planted >= target_count:
            break
        stats = f.get("statistics")
        if stats is None:
            f["statistics"] = {"symmetry_violation": True}
        elif isinstance(stats, dict):
            stats["symmetry_violation"] = True
        planted += 1
    _write_json(char_path, char)


def _mut_plant_metadata_anomaly(bundle_dir: Path, *, run_ids: tuple[str, ...] = ()) -> None:
    """Metadata-axis: plant a metadata_anomaly finding.

    Adds a synthetic Finding with type=trajectory_pattern and
    statistics.pattern_kind='metadata_anomaly' so the metadata-axis
    rule has something to anchor against. If run_ids is empty, uses
    the run_ids from the first existing finding.
    """
    char_path = bundle_dir / "characterization" / "characterization.json"
    char = _read_json(char_path)

    findings = char.get("findings") or []
    if not findings:
        return

    template = findings[0]
    schema_uuid = template["finding_id"].split(":", 1)[0]
    new_id = f"{schema_uuid}:F-EVAL-META"
    used_runs = list(run_ids) or list(template.get("run_ids") or [])

    new_finding = {
        "finding_id": new_id,
        "type": "trajectory_pattern",
        "severity": "major",
        "summary": (
            "Instrument-change confound: bioreactor sensor swapped at t=72h"
            f" across {', '.join(used_runs) or 'multiple'} runs — affects"
            " any cross-run comparison."
        ),
        "confidence": 0.85,
        "extracted_via": "llm_judged",
        "caveats": ["planted metadata anomaly for E2 eval"],
        "competing_explanations": [],
        "evidence_strength": {"n_observations": 1, "n_independent_runs": len(used_runs), "statistical_power": None},
        "evidence_observation_ids": list(template.get("evidence_observation_ids") or [])[:1] or ["planted-obs-eval-meta"],
        "variables_involved": [],
        "time_window": {"start": 72.0, "end": 72.0},
        "run_ids": used_runs,
        "statistics": {"pattern_kind": "metadata_anomaly", "anomaly_kind": "instrument-change"},
    }
    char["findings"] = [new_finding] + findings  # prepend so it's salient
    _write_json(char_path, char)


_MUTATORS: dict[str, callable] = {
    "strip_findings": _mut_strip_findings,
    "strip_trajectories": _mut_strip_trajectories,
    "drop_narratives": _mut_drop_narratives,
    "drop_metadata_anomalies": _mut_drop_metadata_anomalies,
    "plant_weak_n": _mut_plant_weak_n,
    "plant_symmetry_violation": _mut_plant_symmetry_violation,
    "plant_metadata_anomaly": _mut_plant_metadata_anomaly,
    "noop": _mut_noop,
}


def list_mutators() -> list[str]:
    return sorted(_MUTATORS.keys())
