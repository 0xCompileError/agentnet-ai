import pytest

import agentnet as an


def test_communication_protocol_optimizer_selects_best_direct_protocol() -> None:
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
    optimizer = an.CommunicationProtocolOptimizer()

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda protocol: 10.0
        if protocol.representation == "markdown"
        else 1.0,
    )

    assert result.protocol.mode == "direct"
    assert result.protocol.representation == "markdown"
    assert result.protocol.identifier == "markdown"
    assert result.candidate is result.protocol
    assert result.score == 10.0


def test_communication_protocol_optimizer_applies_representation_constraints() -> None:
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
    optimizer = an.CommunicationProtocolOptimizer(
        constraints=[an.RepresentationConstraint(["json"])]
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda protocol: 10.0
        if protocol.representation == "markdown"
        else 1.0,
    )

    assert result.protocol.representation == "json"
    assert result.constraint_results[0].passed is True
    assert result.metadata["rejected_candidates"] == 1


def test_communication_protocol_optimizer_allows_soft_constraint_violations() -> None:
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
    optimizer = an.CommunicationProtocolOptimizer(
        constraints=[
            an.RepresentationConstraint(
                ["json"],
                kind=an.ConstraintKind.SOFT,
            )
        ]
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda protocol: 10.0
        if protocol.representation == "markdown"
        else 1.0,
    )

    assert result.protocol.representation == "markdown"
    assert result.constraint_results[0].passed is False


def test_communication_protocol_optimizer_rejects_incompatible_interfaces() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])
    optimizer = an.CommunicationProtocolOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No communication protocol"):
        optimizer.optimize(source, target, scorer=lambda protocol: 1.0)


def test_communication_protocol_optimizer_reports_final_candidate_counts() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    optimizer = an.CommunicationProtocolOptimizer(
        constraints=[an.RepresentationConstraint(["json", "markdown"])],
        metadata={"optimizer": "communication-protocol"},
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda protocol: 10.0
        if protocol.representation == "markdown"
        else 1.0,
    )

    assert result.protocol.representation == "markdown"
    assert result.metadata["optimizer"] == "communication-protocol"
    assert result.metadata["evaluated_candidates"] == 2
    assert result.metadata["rejected_candidates"] == 1
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "representation",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_communication_protocol_optimizer_is_exported_from_package_root() -> None:
    assert an.CommunicationProtocol is not None
    assert an.CommunicationProtocolOptimizer is not None
    assert an.CommunicationProtocolOptimizationResult is not None
