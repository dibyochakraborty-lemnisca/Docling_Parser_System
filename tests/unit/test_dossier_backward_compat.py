from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fermdocs.domain.models import (
    ConversionStatus,
    Observation,
    ObservationType,
)


def _obs(value_canonical: dict | None) -> Observation:
    return Observation(
        observation_id=uuid.uuid4(),
        experiment_id="EXP-1",
        file_id=uuid.uuid4(),
        column_name="biomass_g_l",
        raw_header="Biomass (X)",
        observation_type=ObservationType.MEASURED,
        value_raw={"value": 14.2, "type": "float"},
        unit_raw="g/L",
        value_canonical=value_canonical,
        unit_canonical="g/L",
        conversion_status=ConversionStatus.OK,
        source_locator={"format": "csv"},
        mapping_confidence=0.95,
        extraction_confidence=1.0,
        needs_review=False,
        extractor_version="v0.1.0",
        extracted_at=datetime.now(timezone.utc),
    )


def test_old_observation_without_via_serializes_cleanly():
    o = _obs(value_canonical={"value": 14.2, "type": "float"})
    out = o.to_dossier_observation()
    assert out["value"] == 14.2
    assert out["via"] is None
    assert out["normalization"] is None


def test_new_observation_with_via_pint():
    o = _obs(value_canonical={"value": 14.2, "type": "float", "via": "pint"})
    out = o.to_dossier_observation()
    assert out["via"] == "pint"
    assert out["normalization"] is None


def test_new_observation_with_normalization_hint():
    o = _obs(
        value_canonical={
            "value": 14.2, "type": "float", "via": "rule_based",
            "normalization": {
                "action": "use_pint_expr",
                "pint_expr": "g/L",
                "rationale": "unicode superscript replaced",
                "confidence": 0.9,
                "source": "rule_based",
            },
        }
    )
    out = o.to_dossier_observation()
    assert out["via"] == "rule_based"
    assert out["normalization"]["action"] == "use_pint_expr"


def test_observation_with_null_canonical_does_not_crash():
    o = _obs(value_canonical=None)
    out = o.to_dossier_observation()
    assert out["via"] is None
    assert out["normalization"] is None
    assert out["value"] == 14.2  # falls through to value_raw
