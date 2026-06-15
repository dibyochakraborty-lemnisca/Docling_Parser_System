"""Claim guard: reject agent claims that contradict the deterministic facts.

Uses the actual false-claim text from the praaj review as the regression corpus.
"""
from __future__ import annotations

from fermdocs.claim_guard import ClaimFacts, check_claim

_CHANNELS = frozenset({"do_pct_saturation", "substrate_g_l", "product_g_l", "biomass_g_l"})


# --- item 5: false "metric not available" -------------------------------------

def test_rejects_no_do_data_when_do_present():
    facts = ClaimFacts(populated_channels=_CHANNELS)
    v = check_claim("There is no DO data available to evaluate oxygen transfer.", facts)
    assert any(x.code == "false_unavailability" for x in v)


def test_rejects_no_substrate_product_limitation_metrics():
    facts = ClaimFacts(populated_channels=_CHANNELS)
    v = check_claim("No substrate limitation metrics are present in this bundle.", facts)
    assert any(x.code == "false_unavailability" for x in v)


def test_allows_unavailability_for_a_genuinely_absent_channel():
    facts = ClaimFacts(populated_channels=_CHANNELS)  # no OUR/CER channel present
    v = check_claim("No OUR or CER data available, so RQ cannot be computed.", facts)
    assert not any(x.code == "false_unavailability" for x in v)


def test_does_not_fire_on_positive_mention_of_a_channel():
    facts = ClaimFacts(populated_channels=_CHANNELS)
    v = check_claim("DO data shows the run sat at zero throughout.", facts)
    assert not any(x.code == "false_unavailability" for x in v)


# --- item 6 tail: oxygen limitation on an anaerobic process -------------------

def test_rejects_oxygen_bottleneck_when_never_aerobic():
    facts = ClaimFacts(populated_channels=_CHANNELS, anaerobic_operation=True)
    v = check_claim("DO = 0.00% indicates a severe oxygen bottleneck / mass transfer failure.", facts)
    assert any(x.code == "oxygen_limitation_when_anaerobic" for x in v)


def test_allows_oxygen_limitation_when_aerobic():
    facts = ClaimFacts(populated_channels=_CHANNELS, anaerobic_operation=False)
    v = check_claim("The DO crash to 0 mid-run indicates oxygen limitation.", facts)
    assert not any(x.code == "oxygen_limitation_when_anaerobic" for x in v)


# --- item 7: scale confound when scale is constant ----------------------------

def test_rejects_scale_confound_when_scale_constant():
    facts = ClaimFacts(reactor_scale_constant=True)
    v = check_claim("Initial volume 61L vs 100L is a scale/bioreactor confound.", facts)
    assert any(x.code == "scale_confound_when_constant" for x in v)


def test_allows_scale_confound_when_scale_unknown():
    facts = ClaimFacts(reactor_scale_constant=None)
    v = check_claim("This may be a scale confound across reactors.", facts)
    assert not any(x.code == "scale_confound_when_constant" for x in v)


# --- item 4 tail: rate at t=0 -------------------------------------------------

def test_rejects_rate_at_t0_and_states_resolution():
    facts = ClaimFacts(sampling_resolution_h=8.0)
    v = check_claim("mu_max occurs at t=0 with no lag phase.", facts)
    hits = [x for x in v if x.code == "rate_at_t0"]
    assert hits and "0–8h" in hits[0].message


def test_allows_rate_at_a_real_time():
    v = check_claim("mu_max = 0.10 1/h at t=12h.", ClaimFacts())
    assert not any(x.code == "rate_at_t0" for x in v)


def test_negated_oxygen_phrasing_is_not_flagged():
    # the pipeline's own CORRECTED A14 finding must not trip the guard
    facts = ClaimFacts(anaerobic_operation=True)
    txt = ("DO stayed at zero the entire run: consistent with anaerobic operation, "
           "NOT an oxygen-transfer limitation.")
    assert not any(x.code == "oxygen_limitation_when_anaerobic" for x in check_claim(txt, facts))


def test_clean_claim_has_no_violations():
    facts = ClaimFacts(populated_channels=_CHANNELS, anaerobic_operation=True,
                       reactor_scale_constant=True, sampling_resolution_h=8.0)
    v = check_claim("Peak titer reached 150 g/L on run B474; substrate was fully consumed.", facts)
    assert v == []
