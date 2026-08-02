import agentnet as an
from agentnet.core.errors import (
    AgentNetConfigurationError,
    AgentNetError,
    AgentNetExecutionError,
    AgentNetStateError,
    AgentNetValidationError,
)


def test_runtime_errors_share_agentnet_base_class() -> None:
    for error_type in (
        AgentNetConfigurationError,
        AgentNetExecutionError,
        AgentNetStateError,
        AgentNetValidationError,
    ):
        error = error_type("failed")

        assert isinstance(error, AgentNetError)
        assert str(error) == "failed"


def test_runtime_errors_are_exported_from_package_root() -> None:
    assert an.AgentNetError is AgentNetError
    assert an.AgentNetExecutionError is AgentNetExecutionError
