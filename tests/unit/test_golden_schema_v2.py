"""Tests for the v2.0 golden schema, GoldenColumn validators, and SpecsProvider.

Covers:
  - GoldenColumn validators (std_dev >= 0, finite numbers)
  - load_schema() against the production YAML (smoke check on v2.0)
  - DictSpecsProvider.from_schema (skips partial entries, sets provenance)
  - DictSpecsProvider.from_schema_with_overrides (CQ1 layering rules:
      a) field-by-field inheritance, b) dossier-only vars, c) explicit null
      suppression, d) provenance tagging)
"""

from __future__ import annotations

import math

import pytest
import yaml
from pydantic import ValidationError

from fermdocs.domain.golden_schema import load_schema
from fermdocs.domain.models import (
    DataType,
    GoldenColumn,
    GoldenSchema,
)
from fermdocs_characterize.specs import DictSpecsProvider


# ---------- GoldenColumn validators ----------


def _column(**overrides):
    base = {
        "name": "biomass_g_l",
        "description": "biomass concentration",
        "data_type": DataType.FLOAT,
        "canonical_unit": "g/L",
    }
    base.update(overrides)
    return GoldenColumn(**base)


def test_std_dev_negative_rejected():
    with pytest.raises(ValidationError):
        _column(nominal=0.5, std_dev=-0.1)


def test_std_dev_zero_allowed():
    col = _column(nominal=0.5, std_dev=0.0)
    assert col.std_dev == 0.0


def test_nominal_inf_rejected():
    with pytest.raises(ValidationError):
        _column(nominal=math.inf, std_dev=0.1)


def test_std_dev_nan_rejected():
    with pytest.raises(ValidationError):
        _column(nominal=0.5, std_dev=math.nan)


def test_nominal_and_std_dev_optional():
    col = _column()
    assert col.nominal is None
    assert col.std_dev is None


# ---------- Production YAML round-trip ----------


def test_production_yaml_loads_as_v2():
    schema = load_schema()
    assert schema.version == "2.0"
    by_name = schema.by_name()
    expected_variables = {
        "biomass_g_l", "mu_x_max_per_h", "mu_p_max_per_h", "substrate_g_l",
        "dissolved_o2_mg_l", "volume_l", "weight_kg", "ph", "temperature_k",
        "paa_mg_l", "nh3_mg_l", "alpha_kla",
    }
    for var in expected_variables:
        assert var in by_name, f"missing variable {var} in v2 schema"
        col = by_name[var]
        assert col.nominal is not None, f"{var} missing nominal in production YAML"
        assert col.std_dev is not None, f"{var} missing std_dev in production YAML"


def test_production_yaml_identifier_columns_have_no_specs():
    by_name = load_schema().by_name()
    for ident in ("experiment_id", "strain_id", "organism", "product"):
        assert ident in by_name
        assert by_name[ident].nominal is None
        assert by_name[ident].std_dev is None


# ---------- from_schema ----------


def _mini_schema(yaml_text: str) -> GoldenSchema:
    return GoldenSchema.model_validate(yaml.safe_load(yaml_text))


SCHEMA_TWO_VARS = """
version: "2.0"
columns:
  - name: temperature_k
    description: temperature
    data_type: float
    canonical_unit: K
    nominal: 297.0
    std_dev: 0.5
  - name: ph
    description: pH
    data_type: float
    canonical_unit: pH
    nominal: 6.5
    std_dev: 0.1
  - name: experiment_id
    description: id
    data_type: text
"""


def test_from_schema_emits_specs_with_schema_provenance():
    schema = _mini_schema(SCHEMA_TWO_VARS)
    provider = DictSpecsProvider.from_schema(schema)

    spec = provider.get("temperature_k")
    assert spec is not None
    assert spec.nominal == 297.0
    assert spec.std_dev == 0.5
    assert spec.unit == "K"
    assert spec.provenance == "schema"


def test_from_schema_skips_columns_without_full_specs():
    schema = _mini_schema(SCHEMA_TWO_VARS)
    provider = DictSpecsProvider.from_schema(schema)
    assert provider.get("experiment_id") is None


def test_from_schema_partial_specs_are_skipped():
    schema = _mini_schema("""
version: "2.0"
columns:
  - name: only_nominal
    description: x
    data_type: float
    canonical_unit: u
    nominal: 1.0
""")
    provider = DictSpecsProvider.from_schema(schema)
    assert provider.get("only_nominal") is None


# ---------- from_schema_with_overrides ----------


def test_overrides_no_dossier_acts_like_from_schema():
    schema = _mini_schema(SCHEMA_TWO_VARS)
    provider = DictSpecsProvider.from_schema_with_overrides(schema, {})
    spec = provider.get("ph")
    assert spec is not None
    assert spec.provenance == "schema"
    assert spec.nominal == 6.5


def test_overrides_field_by_field_inheritance():
    """CQ1a: dossier overrides one field; others inherit from schema."""
    schema = _mini_schema(SCHEMA_TWO_VARS)
    dossier = {"_specs": {"temperature_k": {"nominal": 300.0}}}
    provider = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
    spec = provider.get("temperature_k")
    assert spec is not None
    assert spec.nominal == 300.0
    assert spec.std_dev == 0.5  # inherited from schema
    assert spec.unit == "K"     # inherited from schema
    assert spec.provenance == "merged"


def test_overrides_dossier_only_variable():
    """CQ1b: dossier introduces a variable absent from schema."""
    schema = _mini_schema(SCHEMA_TWO_VARS)
    dossier = {"_specs": {"custom_var": {"nominal": 42.0, "std_dev": 1.0, "unit": "x"}}}
    provider = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
    spec = provider.get("custom_var")
    assert spec is not None
    assert spec.nominal == 42.0
    assert spec.unit == "x"
    assert spec.provenance == "dossier"


def test_overrides_explicit_null_suppresses():
    """CQ1c: setting a field to null in the dossier clears it."""
    schema = _mini_schema(SCHEMA_TWO_VARS)
    dossier = {"_specs": {"temperature_k": {"std_dev": None}}}
    provider = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
    # std_dev cleared -> spec cannot be emitted
    assert provider.get("temperature_k") is None


def test_overrides_carry_source_ref():
    schema = _mini_schema(SCHEMA_TWO_VARS)
    dossier = {
        "_specs": {
            "temperature_k": {"nominal": 300.0, "source_ref": "batch_sheet_42"}
        }
    }
    provider = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
    spec = provider.get("temperature_k")
    assert spec is not None
    assert spec.source_ref == "batch_sheet_42"


def test_overrides_dossier_partial_without_required_fields_falls_back_to_schema():
    """Override that touches only `unit` keeps schema's nominal/std_dev."""
    schema = _mini_schema(SCHEMA_TWO_VARS)
    dossier = {"_specs": {"ph": {"unit": "pH-actual"}}}
    provider = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
    spec = provider.get("ph")
    assert spec is not None
    assert spec.nominal == 6.5
    assert spec.std_dev == 0.1
    assert spec.unit == "pH-actual"
    assert spec.provenance == "merged"


def test_production_yaml_via_overrides_with_empty_dossier():
    """Smoke: real production schema works through the override path."""
    schema = load_schema()
    provider = DictSpecsProvider.from_schema_with_overrides(schema, {})
    for var in ("biomass_g_l", "ph", "alpha_kla"):
        spec = provider.get(var)
        assert spec is not None
        assert spec.provenance == "schema"


# ---------- schema_version on Observation ----------


def test_observation_default_schema_version_is_none():
    from datetime import datetime
    from uuid import uuid4
    from fermdocs.domain.models import Observation, ObservationType, ConversionStatus

    obs = Observation(
        observation_id=uuid4(),
        experiment_id="E1",
        file_id=uuid4(),
        column_name="biomass_g_l",
        raw_header="X",
        observation_type=ObservationType.MEASURED,
        value_raw={"value": 0.5, "type": "float"},
        source_locator={},
        conversion_status=ConversionStatus.OK,
        extractor_version="v0.1.0",
        extracted_at=datetime.utcnow(),
    )
    assert obs.schema_version is None


def test_observation_carries_schema_version_when_set():
    from datetime import datetime
    from uuid import uuid4
    from fermdocs.domain.models import Observation, ObservationType, ConversionStatus

    obs = Observation(
        observation_id=uuid4(),
        experiment_id="E1",
        file_id=uuid4(),
        column_name="biomass_g_l",
        raw_header="X",
        observation_type=ObservationType.MEASURED,
        value_raw={"value": 0.5, "type": "float"},
        source_locator={},
        conversion_status=ConversionStatus.OK,
        extractor_version="v0.1.0",
        schema_version="2.0",
        extracted_at=datetime.utcnow(),
    )
    assert obs.schema_version == "2.0"
