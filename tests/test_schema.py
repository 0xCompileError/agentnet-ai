import pytest

import agentnet as an


def test_schema_validates_mapping_fields() -> None:
    schema = an.Schema({"risks": list[str], "summary": str})
    value = {"risks": ["latency"], "summary": "Looks viable."}

    assert schema.validate(value) == value


def test_schema_rejects_missing_or_wrong_fields() -> None:
    schema = an.Schema({"risks": list[str], "summary": str})

    with pytest.raises(an.AgentNetValidationError, match="value.summary"):
        schema.validate({"risks": ["latency"]})

    with pytest.raises(an.AgentNetValidationError, match="value.risks"):
        schema.validate({"risks": ["latency", 3], "summary": "Looks viable."})


def test_schema_is_exported_from_package_root() -> None:
    from agentnet.core.schema import Schema

    assert an.Schema is Schema
