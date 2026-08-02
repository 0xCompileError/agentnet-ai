import agentnet as an
from agentnet.constraints import SchemaConstraint


def test_schema_constraint_passes_when_candidate_matches_schema() -> None:
    constraint = SchemaConstraint(an.Schema({"summary": str, "risks": list[str]}))

    result = constraint.evaluate({"summary": "ok", "risks": ["latency"]})

    assert result.passed is True
    assert result.blocks_candidate is False


def test_schema_constraint_fails_when_candidate_violates_schema() -> None:
    constraint = SchemaConstraint(an.Schema({"summary": str, "risks": list[str]}))

    result = constraint.evaluate({"summary": "ok", "risks": ["latency", 3]})

    assert result.passed is False
    assert result.blocks_candidate is True


def test_schema_constraint_accepts_type_annotations() -> None:
    constraint = SchemaConstraint(list[str])

    assert constraint.evaluate(["a", "b"]).passed is True
    assert constraint.evaluate(["a", 3]).passed is False


def test_schema_constraint_is_exported_from_package_root() -> None:
    assert an.SchemaConstraint is SchemaConstraint
