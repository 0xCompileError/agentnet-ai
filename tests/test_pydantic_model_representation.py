import pytest

import agentnet as an


class FakePydanticModel:
    def __init__(self, summary: str) -> None:
        self.summary = summary

    @classmethod
    def model_validate(cls, value: object) -> "FakePydanticModel":
        if not isinstance(value, dict) or not isinstance(value.get("summary"), str):
            raise ValueError("invalid summary")
        return cls(value["summary"])


class FakePydanticV1Model:
    def __init__(self, summary: str) -> None:
        self.summary = summary

    @classmethod
    def parse_obj(cls, value: object) -> "FakePydanticV1Model":
        if not isinstance(value, dict) or not isinstance(value.get("summary"), str):
            raise ValueError("invalid summary")
        return cls(value["summary"])


def test_pydantic_model_representation_validates_with_model_validate() -> None:
    representation = an.PydanticModelRepresentation(FakePydanticModel)

    result = representation.validate({"summary": "ok"})

    assert representation.identifier == "pydantic_model"
    assert isinstance(result, FakePydanticModel)
    assert result.summary == "ok"


def test_pydantic_model_representation_validates_with_parse_obj() -> None:
    representation = an.PydanticModelRepresentation(FakePydanticV1Model)

    result = representation.validate({"summary": "ok"})

    assert isinstance(result, FakePydanticV1Model)
    assert result.summary == "ok"


def test_pydantic_model_representation_rejects_invalid_values() -> None:
    representation = an.PydanticModelRepresentation(FakePydanticModel)

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate({"summary": 3}, label="payload")


def test_pydantic_model_representation_rejects_non_model_classes() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="Pydantic"):
        an.PydanticModelRepresentation(object)


def test_pydantic_model_representation_is_exported_from_package_root() -> None:
    assert an.PydanticModelRepresentation is not None
