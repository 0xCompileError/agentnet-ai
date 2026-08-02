import pytest

import agentnet as an


def test_validate_interface_compatibility_accepts_semantics_and_representation() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
        representations=[an.Representation("json"), an.Representation("markdown")],
    )
    target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("markdown")],
    )

    result = an.validate_interface_compatibility(source, target)

    assert result.source is source
    assert result.target is target
    assert result.negotiation.identifier == "markdown"
    assert result.compatible is True


def test_validate_interface_compatibility_rejects_missing_semantics() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
        representations=[an.Representation("json")],
    )

    with pytest.raises(an.AgentNetValidationError, match="risks"):
        an.validate_interface_compatibility(source, target)


def test_validate_interface_compatibility_rejects_incompatible_representations() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])

    with pytest.raises(an.AgentNetValidationError, match="No compatible representation"):
        an.validate_interface_compatibility(source, target)


def test_validate_interface_compatibility_honors_preferred_representation_order() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )

    result = an.validate_interface_compatibility(
        source,
        target,
        preferred=["markdown", "json"],
    )

    assert result.negotiation.identifier == "markdown"


def test_validate_interface_compatibility_treats_missing_representations_as_wildcard() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface()

    result = an.validate_interface_compatibility(source, target)

    assert result.negotiation.identifier == "json"
