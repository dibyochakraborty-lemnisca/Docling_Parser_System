from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fermdocs.domain.models import (
    ConversionStatus,
    NarrativeExtraction,
    Observation,
    ObservationType,
)
from fermdocs.mapping.narrative_extractor import is_dup_of_table_observations


def _table_obs(column: str, value: float) -> Observation:
    return Observation(
        observation_id=uuid.uuid4(),
        experiment_id="E",
        file_id=uuid.uuid4(),
        column_name=column,
        raw_header="Titer (g/L)",
        observation_type=ObservationType.MEASURED,
        value_raw={"value": value, "type": "float"},
        unit_raw="g/L",
        value_canonical={"value": value, "type": "float", "via": "pint"},
        unit_canonical="g/L",
        conversion_status=ConversionStatus.OK,
        source_locator={"format": "pdf", "section": "table", "page": 1, "table_idx": 0},
        mapping_confidence=0.95,
        extraction_confidence=1.0,
        needs_review=False,
        extractor_version="v0.1.0",
        extracted_at=datetime.now(timezone.utc),
    )


def _ext(column: str, value: float) -> NarrativeExtraction:
    return NarrativeExtraction(
        column=column,
        value=value,
        unit="g/L",
        evidence="yield was 14.2 g/L",
        source_paragraph_idx=0,
        confidence=0.9,
    )


def test_exact_value_match_is_dup():
    table = [_table_obs("biomass_g_l", 14.2)]
    assert is_dup_of_table_observations(_ext("biomass_g_l", 14.2), table)


def test_close_value_match_is_dup():
    table = [_table_obs("biomass_g_l", 14.2)]
    assert is_dup_of_table_observations(_ext("biomass_g_l", 14.20001), table)


def test_different_value_not_dup():
    table = [_table_obs("biomass_g_l", 14.2)]
    assert not is_dup_of_table_observations(_ext("biomass_g_l", 13.6), table)


def test_different_column_not_dup():
    table = [_table_obs("biomass_g_l", 14.2)]
    assert not is_dup_of_table_observations(_ext("substrate_g_l", 14.2), table)


def test_no_table_observations_not_dup():
    assert not is_dup_of_table_observations(_ext("biomass_g_l", 14.2), [])


def test_narrative_observation_does_not_dedup_against_self():
    """Only table observations count for dedup, not other narrative ones."""
    narr_obs = _table_obs("biomass_g_l", 14.2)
    narr_obs.source_locator = {"format": "pdf", "section": "narrative"}
    assert not is_dup_of_table_observations(_ext("biomass_g_l", 14.2), [narr_obs])
