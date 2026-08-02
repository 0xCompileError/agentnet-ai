"""AgentNet runtime error hierarchy."""


class AgentNetError(Exception):
    """Base class for AgentNet errors."""


class AgentNetConfigurationError(AgentNetError):
    """Raised when configuration is invalid."""


class AgentNetExecutionError(AgentNetError):
    """Raised when execution fails."""


class AgentNetStateError(AgentNetError):
    """Raised when runtime state is invalid."""


class AgentNetValidationError(AgentNetError):
    """Raised when validation fails."""
