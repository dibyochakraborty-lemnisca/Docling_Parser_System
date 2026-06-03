"""data_feed tests. classify/summarize/leave_one_run_out are JAX-free;
build_feed lazily imports brewtwin."""

import numpy as np
import pandas as pd

from fermdocs_recommend import data_feed


def _df():
    rows = []
    for run in ("RUN-0001", "RUN-0002"):
        for t in (0.0, 1.0, 2.0):
            rows.append((run, "biomass_g_l", t, 1.0 + t, 0, "g/L"))
            rows.append((run, "agitation_rpm", t, 100.0, 0, "rpm"))
            rows.append((run, "feed_rate_l_per_h", t, 0.5, 0, "L/h"))
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value", "imputed", "unit"])


def test_classify_variables():
    cls = data_feed.classify_variables(
        ["biomass_g_l", "agitation_rpm", "feed_rate_l_per_h", "temperature_k"]
    )
    assert "biomass_g_l" in cls["states"]
    assert "agitation_rpm" in cls["controls"]
    assert "temperature_k" in cls["controls"]
    assert "feed_rate_l_per_h" in cls["feed_candidates"]


def test_detect_feed_var():
    assert data_feed.detect_feed_var(["biomass_g_l", "feed_rate_l_per_h"]) == "feed_rate_l_per_h"
    assert data_feed.detect_feed_var(["biomass_g_l", "ph"]) is None


def test_leave_one_run_out():
    assert data_feed.leave_one_run_out(["RUN-0002", "RUN-0001"]) == (["RUN-0001"], "RUN-0002")
    assert data_feed.leave_one_run_out(["RUN-0001"]) == (["RUN-0001"], "RUN-0001")


def test_summarize():
    s = data_feed.summarize(_df())
    assert s["n_runs"] == 2
    assert s["feed_var"] == "feed_rate_l_per_h"
    assert "biomass_g_l" in s["states"]
    assert s["point_counts"]["biomass_g_l"]["real_points"] == 6


def test_get_real_observations_filters_imputed():
    df = pd.DataFrame(
        {
            "run_id": ["R1", "R1", "R1"],
            "variable": ["X", "X", "X"],
            "time_h": [0, 1, 2],
            "value": [1.0, 2.0, 3.0],
            "imputed": [0, 1, 0],  # middle point imputed -> excluded
        }
    )
    t, y = data_feed.get_real_observations(df, "R1", ["X"])
    assert list(t) == [0, 2]
    np.testing.assert_array_equal(y[:, 0], [1.0, 3.0])


def test_build_feed_roundtrip():
    trajs = data_feed.build_feed(_df(), feed_var="feed_rate_l_per_h")
    assert set(trajs) == {"RUN-0001", "RUN-0002"}
    t, v = trajs["RUN-0001"].species_series("biomass_g_l")
    np.testing.assert_array_equal(np.asarray(t), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(v), [1.0, 2.0, 3.0])
