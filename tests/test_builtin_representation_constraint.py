import agentnet as an
from agentnet.constraints import RepresentationConstraint


class RepresentationConfig:
    def __init__(self, representation: str) -> None:
        self.representation = representation


def test_representation_constraint_accepts_allowed_string() -> None:
    constraint = RepresentationConstraint(allowed=["markdown", "plain_text"])

    assert constraint.evaluate("markdown").passed is True
    assert constraint.evaluate("xml").passed is False


def test_representation_constraint_reads_mapping_field() -> None:
    constraint = RepresentationConstraint(allowed=["json"], field="format")

    assert constraint.evaluate({"format": "json"}).passed is True
    assert constraint.evaluate({"format": "yaml"}).passed is False


def test_representation_constraint_reads_object_attribute() -> None:
    constraint = RepresentationConstraint(allowed=["bullet_list"])

    assert constraint.evaluate(RepresentationConfig("bullet_list")).passed is True
    assert constraint.evaluate(RepresentationConfig("xml")).passed is False


def test_representation_constraint_is_exported_from_package_root() -> None:
    assert an.RepresentationConstraint is RepresentationConstraint
