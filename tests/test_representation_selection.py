import pytest

import agentnet as an


def test_select_representation_chooses_highest_scored_compatible_representation() -> None:
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

    selection = an.select_representation(
        source,
        target,
        scorer=lambda representation: 10.0
        if representation.identifier == "markdown"
        else 1.0,
    )

    assert selection.source is source
    assert selection.target is target
    assert selection.representation is not None
    assert selection.representation.identifier == "markdown"
    assert selection.score == 10.0


def test_select_representation_defaults_to_negotiation_order() -> None:
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

    selection = an.select_representation(source, target)

    assert selection.representation is not None
    assert selection.representation.identifier == "json"
    assert selection.score == 0.0


def test_select_representation_rejects_incompatible_interfaces() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])

    with pytest.raises(an.AgentNetValidationError, match="No compatible representation"):
        an.select_representation(source, target)
