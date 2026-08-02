import agentnet as an


def test_detect_incompatible_interfaces_reports_compatible_interfaces() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )

    result = an.detect_incompatible_interfaces(source, target)

    assert result.compatible is True
    assert result.incompatible is False
    assert result.reason is None
    assert result.compatibility is not None
    assert result.representation == "json"


def test_detect_incompatible_interfaces_reports_missing_semantics() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
        representations=[an.Representation("json")],
    )

    result = an.detect_incompatible_interfaces(source, target)

    assert result.compatible is False
    assert result.incompatible is True
    assert result.reason is not None
    assert "risks" in result.reason
    assert result.compatibility is None


def test_detect_incompatible_interfaces_reports_representation_mismatch() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])

    result = an.detect_incompatible_interfaces(source, target)

    assert result.incompatible is True
    assert result.reason is not None
    assert "No compatible representation" in result.reason


def test_detect_incompatible_interfaces_honors_preferred_order() -> None:
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

    result = an.detect_incompatible_interfaces(
        source,
        target,
        preferred=["markdown", "json"],
    )

    assert result.compatible is True
    assert result.representation == "markdown"


def test_incompatible_interface_detection_is_exported_from_package_root() -> None:
    assert an.IncompatibleInterfaceDetection is not None
    assert an.detect_incompatible_interfaces is not None
