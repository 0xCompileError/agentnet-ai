import pytest

import agentnet as an


def test_negotiate_representation_selects_first_common_representation() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("markdown"),
            an.Representation("plain_text"),
        ]
    )

    negotiation = an.negotiate_representation(source, target)

    assert negotiation.source is source
    assert negotiation.target is target
    assert negotiation.representation is source.get_representation("markdown")
    assert negotiation.identifier == "markdown"


def test_negotiate_representation_honors_preferred_order() -> None:
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

    negotiation = an.negotiate_representation(
        source,
        target,
        preferred=["markdown", "json"],
    )

    assert negotiation.identifier == "markdown"


def test_negotiate_representation_rejects_incompatible_interfaces() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])

    with pytest.raises(an.AgentNetValidationError, match="No compatible representation"):
        an.negotiate_representation(source, target)
