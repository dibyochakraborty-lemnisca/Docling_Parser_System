"""Tool registry for the recommendation ReAct loop.

The agent reads the vendored brewtwin skills (get_skill) and fits + predicts in
the sandbox (execute_python, with float64 auto-enabled and brewtwin + the
vendored metrics/data_feed importable). It computes a build_report per model
family and submits them as candidates; the deterministic rubric (applied by the
agent loop, not here) decides the winner or an honest refusal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fermdocs.bundle import BundleReader
from fermdocs_diagnose.tools_bundle.execute_python import execute_python
from fermdocs_recommend import skill_loader

# brewtwin gotcha C1: float64 must be enabled before any heavy JAX call. We
# prepend it to every sandbox snippet so a fit cannot silently run in float32.
_FLOAT64_PREAMBLE = "import jax\njax.config.update('jax_enable_x64', True)\n"

# brewtwin import + stiff fed-batch fit measured at ~8.6s + ~45s in Phase 0.
_FIT_TIMEOUT_DEFAULT = 480


@dataclass
class _AgentState:
    recommendation_payload: dict | None = None
    submitted: bool = False
    tool_calls: int = 0


@dataclass
class RecommendToolBundle:
    reader: BundleReader
    hypothesis_output_path: Path | None = None
    state: _AgentState = field(default_factory=_AgentState)

    @property
    def _obs_path(self) -> Path:
        return self.reader.dir / "characterization" / "observations.csv"

    def _gate(self, tool_name: str) -> dict | None:
        self.state.tool_calls += 1
        if self.state.submitted and tool_name != "submit_recommendation":
            return {"error": "already_finalized", "tool": tool_name}
        return None

    # --- read tools --------------------------------------------------------
    def get_hypotheses(self) -> dict:
        gated = self._gate("get_hypotheses")
        if gated:
            return gated
        path = self.hypothesis_output_path
        if path is None or not Path(path).exists():
            return {"error": "hypothesis_output.json not found", "hypotheses": []}
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "hypotheses": []}
        finals = data.get("final_hypotheses", [])
        slim = [
            {
                "hyp_id": h.get("hyp_id"),
                "summary": h.get("summary"),
                "affected_variables": h.get("affected_variables", []),
                "actionable_recommendation": h.get("actionable_recommendation"),
                "confidence": h.get("confidence"),
            }
            for h in finals
        ]
        return {"hypotheses": slim, "n": len(slim)}

    def get_data_feed(self) -> dict:
        gated = self._gate("get_data_feed")
        if gated:
            return gated
        if not self._obs_path.exists():
            return {"error": "observations.csv not found"}
        from fermdocs_recommend import data_feed

        try:
            df = pd.read_csv(self._obs_path)
            summary = data_feed.summarize(df)
            summary["observations_csv_path"] = str(self._obs_path)
            train, val = data_feed.leave_one_run_out(summary["run_ids"])
            summary["leave_one_run_out"] = {"train": train, "validate": val}
            return summary
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def get_skill(self, name: str) -> dict:
        gated = self._gate("get_skill")
        if gated:
            return gated
        text = skill_loader.load_skill(name)
        if text is None:
            return {"error": f"unknown skill {name!r}", "available": skill_loader.available_skills()}
        return {"skill": name, "content": text}

    def execute_python(self, code: str, timeout: int = _FIT_TIMEOUT_DEFAULT) -> dict:
        gated = self._gate("execute_python")
        if gated:
            return gated
        result = execute_python(_FLOAT64_PREAMBLE + code, timeout=timeout)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
        }

    # --- terminator --------------------------------------------------------
    def submit_recommendation(self, payload: dict) -> dict:
        self.state.tool_calls += 1
        if self.state.submitted:
            return {"error": "already_submitted"}
        self.state.recommendation_payload = payload
        self.state.submitted = True
        return {"ok": True}

    def dispatch(self) -> dict[str, Any]:
        return {
            "get_hypotheses": self.get_hypotheses,
            "get_data_feed": self.get_data_feed,
            "get_skill": self.get_skill,
            "execute_python": self.execute_python,
            "submit_recommendation": self.submit_recommendation,
        }


def make_recommend_tools(
    reader: BundleReader, hypothesis_output_path: Path | None = None
) -> RecommendToolBundle:
    return RecommendToolBundle(reader=reader, hypothesis_output_path=hypothesis_output_path)
