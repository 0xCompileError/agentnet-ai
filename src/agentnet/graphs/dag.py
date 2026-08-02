"""Directed acyclic graph execution."""

from collections.abc import Mapping, Sequence
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetExecutionError, Module


class DAG(Module):
    """Execute named modules according to directed dependencies."""

    def __init__(
        self,
        *,
        nodes: Mapping[str, Module],
        edges: Mapping[str, Sequence[str]] | None = None,
        name: str = "dag",
    ) -> None:
        if not nodes:
            raise AgentNetConfigurationError("DAG requires at least one node")
        super().__init__(name)
        self.nodes = dict(nodes)
        self.edges = _normalize_edges(self.nodes, edges or {})
        self.predecessors = _build_predecessors(self.nodes, self.edges)
        self.output_nodes = tuple(
            node_name for node_name in self.nodes if not self.edges.get(node_name)
        )

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        outputs: dict[str, Any] = {}
        remaining = set(self.nodes)

        while remaining:
            ready = [
                node_name
                for node_name in self.nodes
                if node_name in remaining
                and all(predecessor in outputs for predecessor in self.predecessors[node_name])
            ]
            if not ready:
                raise AgentNetExecutionError(
                    f"DAG {self.name!r} contains a cycle or unresolved dependency"
                )

            for node_name in ready:
                node_input = self._node_input(node_name, input, outputs)
                outputs[node_name] = await self.nodes[node_name].arun(node_input, context)
                remaining.remove(node_name)

        if len(self.output_nodes) == 1:
            return outputs[self.output_nodes[0]]
        return {node_name: outputs[node_name] for node_name in self.output_nodes}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["edges"] = {source: list(targets) for source, targets in self.edges.items()}
        state["nodes"] = {name: node.state_dict() for name, node in self.nodes.items()}
        return state

    def _node_input(
        self,
        node_name: str,
        graph_input: Any,
        outputs: Mapping[str, Any],
    ) -> Any:
        predecessors = self.predecessors[node_name]
        if not predecessors:
            return graph_input
        if len(predecessors) == 1:
            return outputs[predecessors[0]]
        return tuple(outputs[predecessor] for predecessor in predecessors)


def _normalize_edges(
    nodes: Mapping[str, Module],
    edges: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized = {node_name: tuple(edges.get(node_name, ())) for node_name in nodes}
    for source, targets in edges.items():
        if source not in nodes:
            raise AgentNetConfigurationError(f"DAG edge source {source!r} is not a node")
        for target in targets:
            if target not in nodes:
                raise AgentNetConfigurationError(f"DAG edge target {target!r} is not a node")
        normalized[source] = tuple(targets)
    return normalized


def _build_predecessors(
    nodes: Mapping[str, Module],
    edges: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    predecessors: dict[str, list[str]] = {node_name: [] for node_name in nodes}
    for source, targets in edges.items():
        for target in targets:
            predecessors[target].append(source)
    return {node_name: tuple(names) for node_name, names in predecessors.items()}
