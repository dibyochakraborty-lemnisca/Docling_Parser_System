"""Import-guard: the data-native optimizer and the optimization debate must never
(transitively) import the LABS benchmark backend or its Gen-1 scaffolding.

This locks in the de-LABS boundary (2026-06-16): real uploaded data always wins on
the API path, and the data path must not even *load* the synthetic LABS simulator.
Runs in a fresh subprocess so other tests' imports can't pollute sys.modules.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_data_path_and_debate_import_no_labs():
    probe = textwrap.dedent(
        """
        import sys
        # The data-native optimizer + the opportunity debate.
        import fermdocs_optimize.data_equation
        import fermdocs_optimize.lever_discovery
        import fermdocs_optimize.discovery.general_mech
        import fermdocs_optimize_debate.loader
        import fermdocs_optimize_debate.topics
        import fermdocs_optimize_debate.schema

        forbidden_substrings = (
            "fermdocs_optimize.benchmark",      # the LABS simulator package
            "models.mechanistic",               # Gen-1 X/S/P/M/V model
            "fermdocs_optimize.active_optimize", # Gen-1 active learning
            "fermdocs_optimize.oracle_search",
            "fermdocs_optimize.scipy_search",
            "fermdocs_optimize.discovery.loop",  # Gen-1 LABS discovery loop
            "fermdocs_optimize.discovery.candidate_model",
        )
        leaked = sorted(
            m for m in sys.modules
            if any(s in m for s in forbidden_substrings)
        )
        if leaked:
            print("LEAK:" + ",".join(leaked))
            sys.exit(1)
        print("CLEAN")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        "data path / debate transitively imports LABS or Gen-1 scaffolding:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "CLEAN" in proc.stdout
