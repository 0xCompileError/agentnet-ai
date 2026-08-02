"""Built-in constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import Any, cast

from agentnet.constraints.base import (
    Constraint,
    ConstraintDescriptor,
    ConstraintKind,
    ConstraintResult,
)
from agentnet.core import AgentNetConfigurationError, AgentNetValidationError, Module
from agentnet.core.schema import validate_schema
from agentnet.graphs import CompiledGraph, validate_graph
from agentnet.llms import LLMPolicy, ModelRef
from agentnet.policies import RetryPolicy
from agentnet.tools import ToolRegistry, ToolSpec


class CustomConstraint(Constraint):
    """Constraint backed by an explicit in-process predicate."""

    def __init__(
        self,
        name: str,
        predicate: object,
        *,
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not callable(predicate):
            raise AgentNetConfigurationError(
                "CustomConstraint predicate must be callable"
            )
        self.predicate = cast(Callable[[Any, Any | None], bool], predicate)
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return bool(self.predicate(candidate, context))

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={"custom": True},
        )


class CostConstraint(Constraint):
    """Constraint requiring cost to stay at or below a maximum."""

    def __init__(
        self,
        max_cost: float,
        *,
        field: str = "cost",
        name: str = "cost",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if max_cost < 0:
            raise AgentNetConfigurationError("CostConstraint max_cost cannot be negative")
        self.max_cost = float(max_cost)
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        cost = _extract_numeric_value(candidate, self.field)
        return cost is not None and cost <= self.max_cost

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "field": self.field,
                "max_cost": self.max_cost,
            },
        )


class TokenConstraint(Constraint):
    """Constraint requiring token count to stay at or below a maximum."""

    def __init__(
        self,
        max_tokens: int,
        *,
        field: str = "tokens",
        name: str = "tokens",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if max_tokens < 0:
            raise AgentNetConfigurationError(
                "TokenConstraint max_tokens cannot be negative"
            )
        self.max_tokens = int(max_tokens)
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        tokens = _extract_numeric_value(candidate, self.field)
        return tokens is not None and tokens <= self.max_tokens

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "field": self.field,
                "max_tokens": self.max_tokens,
            },
        )


class LatencyConstraint(Constraint):
    """Constraint requiring latency to stay at or below a maximum."""

    def __init__(
        self,
        max_latency_ms: float,
        *,
        field: str = "latency_ms",
        name: str = "latency",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if max_latency_ms < 0:
            raise AgentNetConfigurationError(
                "LatencyConstraint max_latency_ms cannot be negative"
            )
        self.max_latency_ms = float(max_latency_ms)
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        latency_ms = _extract_numeric_value(candidate, self.field)
        return latency_ms is not None and latency_ms <= self.max_latency_ms

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "field": self.field,
                "max_latency_ms": self.max_latency_ms,
            },
        )


class MemoryConstraint(Constraint):
    """Constraint requiring memory usage to stay at or below a maximum."""

    def __init__(
        self,
        max_memory_mb: float,
        *,
        field: str = "memory_mb",
        name: str = "memory",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if max_memory_mb < 0:
            raise AgentNetConfigurationError(
                "MemoryConstraint max_memory_mb cannot be negative"
            )
        self.max_memory_mb = float(max_memory_mb)
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        memory_mb = _extract_numeric_value(candidate, self.field)
        return memory_mb is not None and memory_mb <= self.max_memory_mb

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "field": self.field,
                "max_memory_mb": self.max_memory_mb,
            },
        )


class SafetyConstraint(Constraint):
    """Constraint requiring text not to contain blocked terms."""

    def __init__(
        self,
        blocked_terms: Iterable[str],
        *,
        field: str = "content",
        case_sensitive: bool = False,
        name: str = "safety",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        terms = tuple(str(term) for term in blocked_terms)
        if not terms:
            raise AgentNetConfigurationError(
                "SafetyConstraint requires at least one blocked term"
            )
        self.blocked_terms = terms
        self.field = field
        self.case_sensitive = case_sensitive
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        text = _extract_text_value(candidate, self.field)
        if text is None:
            return False
        haystack = text if self.case_sensitive else text.lower()
        terms = self.blocked_terms if self.case_sensitive else tuple(
            term.lower() for term in self.blocked_terms
        )
        return not any(term in haystack for term in terms)

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "blocked_terms": list(self.blocked_terms),
                "case_sensitive": self.case_sensitive,
                "field": self.field,
            },
        )


class TopologyConstraint(Constraint):
    """Constraint requiring graph topology to stay within structural bounds."""

    def __init__(
        self,
        *,
        allowed_modules: Iterable[str] | None = None,
        max_nodes: int | None = None,
        max_branches: int | None = None,
        max_depth: int | None = None,
        name: str = "topology",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        self.allowed_modules = (
            None
            if allowed_modules is None
            else tuple(str(value) for value in allowed_modules)
        )
        self.max_nodes = max_nodes
        self.max_branches = max_branches
        self.max_depth = max_depth
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        graph = _extract_compiled_graph(candidate)
        if graph is None:
            return False
        if self.allowed_modules is not None:
            allowed = set(self.allowed_modules)
            if any(
                module.__class__.__name__ not in allowed
                for module in graph.nodes.values()
            ):
                return False
        if self.max_nodes is not None and len(graph.nodes) > self.max_nodes:
            return False
        if self.max_branches is not None:
            max_branches = max((len(targets) for targets in graph.edges.values()), default=0)
            if max_branches > self.max_branches:
                return False
        if self.max_depth is not None and _compiled_graph_depth(graph) > self.max_depth:
            return False
        return True

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "allowed_modules": (
                    None
                    if self.allowed_modules is None
                    else list(self.allowed_modules)
                ),
                "max_branches": self.max_branches,
                "max_depth": self.max_depth,
                "max_nodes": self.max_nodes,
            },
        )


class RetryConstraint(Constraint):
    """Constraint requiring retry policy settings to stay within bounds."""

    def __init__(
        self,
        *,
        max_transport_retries: int | None = None,
        max_quality_retries: int | None = None,
        max_total_attempts: int | None = None,
        allowed_backoff: Iterable[str] | None = None,
        name: str = "retry",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        self.max_transport_retries = max_transport_retries
        self.max_quality_retries = max_quality_retries
        self.max_total_attempts = max_total_attempts
        self.allowed_backoff = (
            None if allowed_backoff is None else tuple(str(value) for value in allowed_backoff)
        )
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        policy = _extract_retry_policy(candidate)
        if policy is None:
            return False
        if (
            self.max_transport_retries is not None
            and policy.transport_retries > self.max_transport_retries
        ):
            return False
        if (
            self.max_quality_retries is not None
            and policy.quality_retries > self.max_quality_retries
        ):
            return False
        if (
            self.max_total_attempts is not None
            and (
                policy.max_total_attempts is None
                or policy.max_total_attempts > self.max_total_attempts
            )
        ):
            return False
        return self.allowed_backoff is None or policy.backoff in self.allowed_backoff

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "allowed_backoff": (
                    None
                    if self.allowed_backoff is None
                    else list(self.allowed_backoff)
                ),
                "max_quality_retries": self.max_quality_retries,
                "max_total_attempts": self.max_total_attempts,
                "max_transport_retries": self.max_transport_retries,
            },
        )


class ToolConstraint(Constraint):
    """Constraint requiring all referenced tools to be allowed."""

    def __init__(
        self,
        allowed: Iterable[str],
        *,
        field: str = "tools",
        name: str = "tool",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        allowed_values = tuple(str(value) for value in allowed)
        if not allowed_values:
            raise AgentNetConfigurationError(
                "ToolConstraint requires at least one allowed tool"
            )
        self.allowed = allowed_values
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        tools = _extract_tool_names(candidate, self.field)
        return bool(tools) and all(tool in self.allowed for tool in tools)

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "allowed": list(self.allowed),
                "field": self.field,
            },
        )


class ModelConstraint(Constraint):
    """Constraint requiring all referenced models to be allowed."""

    def __init__(
        self,
        allowed: Iterable[str],
        *,
        field: str = "model",
        name: str = "model",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        allowed_values = tuple(str(value) for value in allowed)
        if not allowed_values:
            raise AgentNetConfigurationError(
                "ModelConstraint requires at least one allowed model"
            )
        self.allowed = allowed_values
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        models = _extract_model_aliases(candidate, self.field)
        return bool(models) and all(model in self.allowed for model in models)

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "allowed": list(self.allowed),
                "field": self.field,
            },
        )


class RepresentationConstraint(Constraint):
    """Constraint requiring an allowed representation identifier."""

    def __init__(
        self,
        allowed: Iterable[str],
        *,
        field: str = "representation",
        name: str = "representation",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        allowed_values = tuple(str(value) for value in allowed)
        if not allowed_values:
            raise AgentNetConfigurationError(
                "RepresentationConstraint requires at least one allowed value"
            )
        self.allowed = allowed_values
        self.field = field
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        representation = _extract_named_value(candidate, self.field)
        return representation is not None and str(representation) in self.allowed

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "allowed": list(self.allowed),
                "field": self.field,
            },
        )


class SchemaConstraint(Constraint):
    """Constraint requiring a candidate to satisfy a schema."""

    def __init__(
        self,
        schema: Any,
        *,
        name: str = "schema",
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        label: str = "candidate",
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        self.schema = schema
        self.label = label
        super().__init__(
            name,
            description=description,
            kind=kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        try:
            validate_schema(self.schema, candidate, label=self.label)
        except AgentNetValidationError:
            return False
        return True

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        try:
            validate_schema(self.schema, candidate, label=self.label)
        except AgentNetValidationError as exc:
            return ConstraintResult(
                constraint=self.name,
                passed=False,
                kind=self.kind,
                message=str(exc),
            )
        return ConstraintResult(
            constraint=self.name,
            passed=True,
            kind=self.kind,
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={
                "label": self.label,
                "schema": repr(self.schema),
            },
        )


def _extract_named_value(candidate: Any, field: str) -> Any | None:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, Mapping):
        return candidate.get(field)
    return getattr(candidate, field, None)


def _extract_model_aliases(candidate: Any, field: str) -> tuple[str, ...]:
    if isinstance(candidate, str):
        return (candidate,)
    if isinstance(candidate, ModelRef):
        return (candidate.alias,)
    if isinstance(candidate, LLMPolicy):
        return candidate.candidates
    if hasattr(candidate, "llms"):
        return tuple(_model_alias(model) for model in candidate.llms)

    value = _extract_named_value(candidate, field)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("models")
    if value is None:
        return ()
    if isinstance(value, str | ModelRef):
        return (_model_alias(value),)
    if isinstance(value, Iterable):
        return tuple(_model_alias(model) for model in value)
    return (_model_alias(value),)


def _model_alias(model: Any) -> str:
    if isinstance(model, ModelRef):
        return model.alias
    if hasattr(model, "alias"):
        return str(model.alias)
    return str(model)


def _extract_tool_names(candidate: Any, field: str) -> tuple[str, ...]:
    if isinstance(candidate, str):
        return (candidate,)
    if isinstance(candidate, ToolSpec):
        return (candidate.name,)
    if isinstance(candidate, ToolRegistry):
        return candidate.names
    if hasattr(candidate, "allowed_tools"):
        return tuple(str(tool) for tool in candidate.allowed_tools)
    if hasattr(candidate, "tools"):
        return tuple(str(tool) for tool in candidate.tools)

    value = _extract_named_value(candidate, field)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("tool")
    if value is None:
        return ()
    if isinstance(value, str | ToolSpec):
        return (_tool_name(value),)
    if isinstance(value, Iterable):
        return tuple(_tool_name(tool) for tool in value)
    return (_tool_name(value),)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, ToolSpec):
        return tool.name
    if hasattr(tool, "name"):
        return str(tool.name)
    return str(tool)


def _extract_retry_policy(candidate: Any) -> RetryPolicy | None:
    if isinstance(candidate, RetryPolicy):
        return candidate
    if hasattr(candidate, "retry_policy"):
        value = candidate.retry_policy
        return value if isinstance(value, RetryPolicy) else None
    if isinstance(candidate, Mapping):
        value = candidate.get("retry_policy", candidate)
        if isinstance(value, RetryPolicy):
            return value
        if isinstance(value, Mapping):
            return RetryPolicy.from_dict(dict(value))
    return None


def _extract_numeric_value(candidate: Any, field: str) -> float | None:
    if isinstance(candidate, Real) and not isinstance(candidate, bool):
        return float(candidate)
    value = _extract_named_value(candidate, field)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return None


def _extract_text_value(candidate: Any, field: str) -> str | None:
    if isinstance(candidate, str):
        return candidate
    value = _extract_named_value(candidate, field)
    if value is None:
        return None
    return str(value)


def _extract_compiled_graph(candidate: Any) -> CompiledGraph | None:
    if isinstance(candidate, CompiledGraph):
        return candidate
    if isinstance(candidate, Module):
        try:
            return validate_graph(candidate)
        except AgentNetValidationError:
            return None
    return None


def _compiled_graph_depth(graph: CompiledGraph) -> int:
    depths: dict[str, int] = {}

    def visit(node: str) -> int:
        if node in depths:
            return depths[node]
        targets = graph.edges.get(node, ())
        depth = 1 if not targets else 1 + max(visit(target) for target in targets)
        depths[node] = depth
        return depth

    return max((visit(node) for node in graph.entry_nodes), default=0)
