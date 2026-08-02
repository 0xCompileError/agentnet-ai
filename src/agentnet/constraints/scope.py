"""Scoped constraint wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentnet.constraints.base import (
    Constraint,
    ConstraintDescriptor,
    ConstraintKind,
    ConstraintResult,
)
from agentnet.core import AgentNetConfigurationError, Module
from agentnet.graphs.compiler import CompiledGraph, compile_graph


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Resolved graph edge passed to edge-level constraints."""

    source: str
    target: str
    source_module: Module
    target_module: Module

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


class NodeConstraint(Constraint):
    """Apply an inner constraint to one named graph node."""

    def __init__(
        self,
        node: str,
        constraint: Constraint,
        *,
        name: str | None = None,
        description: str | None = None,
        kind: ConstraintKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not node:
            raise AgentNetConfigurationError("NodeConstraint node cannot be empty")
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                "NodeConstraint can only wrap a Constraint instance"
            )
        self.node = node
        self.constraint = constraint
        super().__init__(
            name or f"node({node}):{constraint.name}",
            description=description,
            kind=kind or constraint.kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return self.evaluate(candidate, context).passed

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        graph = _compile_candidate(candidate)
        if graph is None or self.node not in graph.nodes:
            return ConstraintResult(
                constraint=self.name,
                passed=False,
                kind=self.kind,
                message=f"Constraint {self.name!r} failed",
                metadata={"node": self.node},
            )

        child_result = self.constraint.evaluate(graph.nodes[self.node], context)
        return ConstraintResult(
            constraint=self.name,
            passed=child_result.passed,
            kind=self.kind,
            message=None if child_result.passed else f"Constraint {self.name!r} failed",
            metadata={
                "node": self.node,
                "result": child_result.to_dict(),
            },
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={"node": self.node},
            children=(self.constraint.to_descriptor(),),
        )


class GraphConstraint(Constraint):
    """Apply an inner constraint to a compiled graph."""

    def __init__(
        self,
        constraint: Constraint,
        *,
        name: str | None = None,
        description: str | None = None,
        kind: ConstraintKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                "GraphConstraint can only wrap a Constraint instance"
            )
        self.constraint = constraint
        super().__init__(
            name or f"graph:{constraint.name}",
            description=description,
            kind=kind or constraint.kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return self.evaluate(candidate, context).passed

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        graph = _compile_candidate(candidate)
        if graph is None:
            return ConstraintResult(
                constraint=self.name,
                passed=False,
                kind=self.kind,
                message=f"Constraint {self.name!r} failed",
                metadata={"graph": None},
            )

        child_result = self.constraint.evaluate(graph, context)
        return ConstraintResult(
            constraint=self.name,
            passed=child_result.passed,
            kind=self.kind,
            message=None if child_result.passed else f"Constraint {self.name!r} failed",
            metadata={"result": child_result.to_dict()},
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            children=(self.constraint.to_descriptor(),),
        )


def _compile_candidate(candidate: Any) -> CompiledGraph | None:
    if isinstance(candidate, CompiledGraph):
        return candidate
    if isinstance(candidate, Module):
        return compile_graph(candidate)
    return None


class EdgeConstraint(Constraint):
    """Apply an inner constraint to one directed graph edge."""

    def __init__(
        self,
        source: str,
        target: str,
        constraint: Constraint,
        *,
        name: str | None = None,
        description: str | None = None,
        kind: ConstraintKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not source or not target:
            raise AgentNetConfigurationError(
                "EdgeConstraint source and target cannot be empty"
            )
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                "EdgeConstraint can only wrap a Constraint instance"
            )
        self.source = source
        self.target = target
        self.constraint = constraint
        super().__init__(
            name or f"edge({source}->{target}):{constraint.name}",
            description=description,
            kind=kind or constraint.kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return self.evaluate(candidate, context).passed

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        graph = _compile_candidate(candidate)
        edge = None if graph is None else _resolve_edge(graph, self.source, self.target)
        edge_metadata = {"edge": {"source": self.source, "target": self.target}}
        if edge is None:
            return ConstraintResult(
                constraint=self.name,
                passed=False,
                kind=self.kind,
                message=f"Constraint {self.name!r} failed",
                metadata=edge_metadata,
            )

        child_result = self.constraint.evaluate(edge, context)
        return ConstraintResult(
            constraint=self.name,
            passed=child_result.passed,
            kind=self.kind,
            message=None if child_result.passed else f"Constraint {self.name!r} failed",
            metadata={
                **edge_metadata,
                "result": child_result.to_dict(),
            },
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={"source": self.source, "target": self.target},
            children=(self.constraint.to_descriptor(),),
        )


def _resolve_edge(
    graph: CompiledGraph, source: str, target: str
) -> GraphEdge | None:
    if source not in graph.nodes or target not in graph.nodes:
        return None
    if target not in graph.edges.get(source, ()):
        return None
    return GraphEdge(
        source=source,
        target=target,
        source_module=graph.nodes[source],
        target_module=graph.nodes[target],
    )
