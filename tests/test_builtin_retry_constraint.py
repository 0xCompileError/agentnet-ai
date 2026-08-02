import agentnet as an
from agentnet.constraints import RetryConstraint


def test_retry_constraint_accepts_policy_within_bounds() -> None:
    constraint = RetryConstraint(
        max_transport_retries=2,
        max_quality_retries=1,
        max_total_attempts=4,
        allowed_backoff=["none", "constant"],
    )

    assert (
        constraint.evaluate(
            an.RetryPolicy(
                transport_retries=2,
                quality_retries=1,
                backoff="constant",
                max_total_attempts=4,
            )
        ).passed
        is True
    )


def test_retry_constraint_rejects_policy_outside_bounds() -> None:
    constraint = RetryConstraint(max_transport_retries=1, allowed_backoff=["none"])

    assert constraint.evaluate(an.RetryPolicy(transport_retries=2)).passed is False
    assert constraint.evaluate(an.RetryPolicy(backoff="exponential")).passed is False


def test_retry_constraint_reads_react_agent_retry_policy() -> None:
    constraint = RetryConstraint(max_quality_retries=1)

    assert (
        constraint.evaluate(
            an.ReActAgent("agent", retry_policy=an.RetryPolicy(quality_retries=1))
        ).passed
        is True
    )
    assert (
        constraint.evaluate(
            an.ReActAgent("agent", retry_policy=an.RetryPolicy(quality_retries=2))
        ).passed
        is False
    )


def test_retry_constraint_is_exported_from_package_root() -> None:
    assert an.RetryConstraint is RetryConstraint
