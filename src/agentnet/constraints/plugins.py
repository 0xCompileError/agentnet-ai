"""Custom constraint plugin registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agentnet.constraints.base import Constraint
from agentnet.core import AgentNetConfigurationError


class ConstraintPluginRegistry:
    """Registry for explicitly provided custom constraint factories."""

    def __init__(
        self,
        factories: Mapping[str, Callable[..., object]] | None = None,
    ) -> None:
        self._factories: dict[str, Callable[..., object]] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(self, name: str, factory: Callable[..., object]) -> None:
        if not name:
            raise AgentNetConfigurationError("Constraint plugin name cannot be empty")
        if name in self._factories:
            raise AgentNetConfigurationError(
                f"Constraint plugin {name!r} is already registered"
            )
        if not callable(factory):
            raise AgentNetConfigurationError(
                "Constraint plugin factory must be callable"
            )
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Constraint:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(
                f"Unknown constraint plugin {name!r}"
            ) from exc

        constraint = factory(**kwargs)
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                f"Constraint plugin {name!r} did not return a Constraint"
            )
        return constraint

    def to_dict(self) -> dict[str, list[str]]:
        return {"plugins": sorted(self._factories)}
