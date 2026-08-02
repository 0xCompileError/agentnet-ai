import pytest

import agentnet as an
from agentnet.llms.policy import LLMPolicy


def test_llm_policy_tracks_primary_and_fallback_models() -> None:
    policy = LLMPolicy(candidates=["strong", "cheap", "backup"])

    assert policy.candidates == ("strong", "cheap", "backup")
    assert policy.primary == "strong"
    assert policy.fallbacks == ("cheap", "backup")


def test_llm_policy_reorders_fallbacks_without_changing_primary() -> None:
    policy = LLMPolicy(candidates=["strong", "cheap", "backup"])

    reordered = policy.with_fallback_order(["backup", "cheap"])

    assert reordered.primary == "strong"
    assert reordered.fallbacks == ("backup", "cheap")
    assert policy.fallbacks == ("cheap", "backup")


def test_llm_policy_rejects_invalid_fallback_order() -> None:
    policy = LLMPolicy(candidates=["strong", "cheap", "backup"])

    with pytest.raises(an.AgentNetConfigurationError, match="fallback"):
        policy.with_fallback_order(["cheap"])

    with pytest.raises(an.AgentNetConfigurationError, match="fallback"):
        policy.with_fallback_order(["cheap", "cheap"])


def test_llm_policy_round_trips_order_and_retry_policy() -> None:
    policy = LLMPolicy(
        candidates=[an.ModelRef("strong"), "cheap", "backup"],
        retry_policy=an.RetryPolicy(transport_retries=1, quality_retries=2),
    ).with_fallback_order(["backup", "cheap"])

    restored = LLMPolicy.from_dict(policy.to_dict())

    assert restored == policy
    assert restored.candidates == ("strong", "backup", "cheap")
    assert restored.retry_policy == an.RetryPolicy(transport_retries=1, quality_retries=2)


def test_llm_policy_rejects_empty_candidates() -> None:
    with pytest.raises(an.AgentNetConfigurationError):
        LLMPolicy(candidates=[])


def test_llm_policy_is_exported_from_package_root() -> None:
    assert an.LLMPolicy is LLMPolicy
