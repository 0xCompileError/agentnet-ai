import agentnet as an
from agentnet.constraints import ModelConstraint


def test_model_constraint_accepts_allowed_model_string_and_model_ref() -> None:
    constraint = ModelConstraint(allowed=["cheap", "strong"])

    assert constraint.evaluate("cheap").passed is True
    assert constraint.evaluate(an.ModelRef("strong")).passed is True
    assert constraint.evaluate("unapproved").passed is False


def test_model_constraint_requires_all_policy_candidates_to_be_allowed() -> None:
    constraint = ModelConstraint(allowed=["cheap", "backup"])

    assert constraint.evaluate(an.LLMPolicy(["cheap", "backup"])).passed is True
    assert constraint.evaluate(an.LLMPolicy(["cheap", "strong"])).passed is False


def test_model_constraint_reads_react_agent_llms() -> None:
    constraint = ModelConstraint(allowed=["cheap", "backup"])

    assert constraint.evaluate(an.ReActAgent("agent", llms=["cheap", "backup"])).passed is True
    assert constraint.evaluate(an.ReActAgent("agent", llms=["cheap", "strong"])).passed is False


def test_model_constraint_is_exported_from_package_root() -> None:
    assert an.ModelConstraint is ModelConstraint
