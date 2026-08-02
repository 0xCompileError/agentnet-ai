import pytest

import agentnet as an
from agentnet.llms.model_ref import ModelRef


def test_model_ref_stores_alias_and_optional_metadata() -> None:
    model_ref = ModelRef("strong", provider="openai", model="gpt-4o")

    assert model_ref.alias == "strong"
    assert model_ref.provider == "openai"
    assert model_ref.model == "gpt-4o"
    assert str(model_ref) == "strong"


def test_model_ref_round_trips_to_dict() -> None:
    model_ref = ModelRef("cheap")

    assert ModelRef.from_dict(model_ref.to_dict()) == model_ref


def test_model_ref_rejects_empty_alias() -> None:
    with pytest.raises(an.AgentNetConfigurationError):
        ModelRef("")


def test_model_ref_is_exported_from_package_root() -> None:
    assert an.ModelRef is ModelRef
