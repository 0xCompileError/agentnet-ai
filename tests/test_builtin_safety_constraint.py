import agentnet as an
from agentnet.constraints import SafetyConstraint


class SafetyRecord:
    def __init__(self, content: str) -> None:
        self.content = content


def test_safety_constraint_blocks_configured_terms_in_text() -> None:
    constraint = SafetyConstraint(blocked_terms=["secret"])

    assert constraint.evaluate("public summary").passed is True
    assert constraint.evaluate("contains SECRET value").passed is False


def test_safety_constraint_reads_mapping_field() -> None:
    constraint = SafetyConstraint(blocked_terms=["secret"], field="output")

    assert constraint.evaluate({"output": "public"}).passed is True
    assert constraint.evaluate({"output": "secret"}).passed is False


def test_safety_constraint_reads_object_attribute() -> None:
    constraint = SafetyConstraint(blocked_terms=["secret"])

    assert constraint.evaluate(SafetyRecord("public")).passed is True
    assert constraint.evaluate(SafetyRecord("secret")).passed is False


def test_safety_constraint_is_exported_from_package_root() -> None:
    assert an.SafetyConstraint is SafetyConstraint
