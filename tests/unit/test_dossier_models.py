from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fermdocs.domain.models import (
    ConversionStatus,
    Observation,
    ObservationType,
)


def test_observation_to_dossier_includes_all_keys():
    o = Observation(
        observation_id=uuid.uuid4(),
        experiment_id="EXP-1",
        file_id=uuid.uuid4(),
        column_name="biomass_g_l",
        raw_header="Biomass (X)",
        observation_type=ObservationType.MEASURED,
        value_raw={"value": 0.5, "type": "float"},
        unit_raw="g/L",
        value_canonical={"value": 0.5, "type": "float"},
        unit_canonical="g/L",
        conversion_status=ConversionStatus.OK,
        source_locator={"format": "csv", "row": 0, "col": 6},
        mapping_confidence=0.95,
        extraction_confidence=1.0,
        needs_review=False,
        extractor_version="v0.1.0",
        extracted_at=datetime.now(timezone.utc),
    )
    out = o.to_dossier_observation()
    assert out["value"] == 0.5
    assert out["unit"] == "g/L"
    assert out["confidence"]["combined"] == 0.95
    assert out["source"]["raw_header"] == "Biomass (X)"
    assert out["source"]["locator"]["row"] == 0
