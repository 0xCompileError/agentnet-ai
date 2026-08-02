import pytest

import agentnet as an


def test_retry_policy_stores_configuration_defensively() -> None:
    fallback_on = ["timeout", "schema_failure"]

    policy = an.RetryPolicy(
        transport_retries=4,
        quality_retries=2,
        fallback_on=fallback_on,
        backoff="linear",
        backoff_base_seconds=0.25,
        max_total_attempts=5,
    )
    fallback_on.append("rate_limit")

    assert policy.transport_retries == 4
    assert policy.quality_retries == 2
    assert policy.fallback_on == ("timeout", "schema_failure")
    assert policy.backoff == "linear"
    assert policy.backoff_base_seconds == 0.25
    assert policy.max_total_attempts == 5


def test_retry_policy_defaults_match_planned_contract() -> None:
    policy = an.RetryPolicy()

    assert policy.transport_retries == 2
    assert policy.quality_retries == 1
    assert policy.fallback_on == ("timeout", "rate_limit", "api_error", "schema_failure")
    assert policy.backoff == "exponential"
    assert policy.backoff_base_seconds == 0.0
    assert policy.max_total_attempts is None


def test_retry_policy_rejects_invalid_configuration() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="transport_retries"):
        an.RetryPolicy(transport_retries=-1)

    with pytest.raises(an.AgentNetConfigurationError, match="quality_retries"):
        an.RetryPolicy(quality_retries=-1)

    with pytest.raises(an.AgentNetConfigurationError, match="fallback_on"):
        an.RetryPolicy(fallback_on=[])

    with pytest.raises(an.AgentNetConfigurationError, match="fallback_on"):
        an.RetryPolicy(fallback_on=["timeout", ""])

    with pytest.raises(an.AgentNetConfigurationError, match="backoff"):
        an.RetryPolicy(backoff="random")

    with pytest.raises(an.AgentNetConfigurationError, match="backoff_base_seconds"):
        an.RetryPolicy(backoff_base_seconds=-0.1)

    with pytest.raises(an.AgentNetConfigurationError, match="max_total_attempts"):
        an.RetryPolicy(max_total_attempts=0)


def test_retry_policy_round_trips_to_dict() -> None:
    policy = an.RetryPolicy(
        transport_retries=1,
        quality_retries=3,
        fallback_on=["timeout", "judge_failure"],
        backoff="constant",
        backoff_base_seconds=0.5,
        max_total_attempts=4,
    )

    restored = an.RetryPolicy.from_dict(policy.to_dict())

    assert restored == policy
    assert restored.fallback_on is not policy.fallback_on


def test_retry_policy_checks_fallback_reasons() -> None:
    policy = an.RetryPolicy(fallback_on=["timeout", "api_error"])

    assert policy.should_fallback("timeout") is True
    assert policy.should_fallback("schema_failure") is False


def test_retry_policy_calculates_backoff_delays() -> None:
    assert an.RetryPolicy(backoff="none", backoff_base_seconds=0.5).backoff_delay(3) == 0.0
    assert an.RetryPolicy(backoff="constant", backoff_base_seconds=0.5).backoff_delay(3) == 0.5
    assert an.RetryPolicy(backoff="linear", backoff_base_seconds=0.5).backoff_delay(3) == 1.5
    assert an.RetryPolicy(backoff="exponential", backoff_base_seconds=0.5).backoff_delay(3) == 2.0

    with pytest.raises(an.AgentNetConfigurationError, match="retry_number"):
        an.RetryPolicy(backoff_base_seconds=0.5).backoff_delay(0)


def test_retry_policy_is_exported_from_package_root() -> None:
    from agentnet.policies import RetryPolicy

    assert an.RetryPolicy is RetryPolicy
