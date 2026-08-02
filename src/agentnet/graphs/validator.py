"""Compiled graph validation."""

from agentnet.core import AgentNetValidationError, Module
from agentnet.graphs.compiler import CompiledGraph, compile_graph
from agentnet.interfaces import Interface, validate_interface_compatibility


class GraphValidator:
    """Validate compiled graph structure."""

    def validate(self, graph: CompiledGraph | Module) -> CompiledGraph:
        compiled = compile_graph(graph) if isinstance(graph, Module) else graph
        _validate_references(compiled)
        _validate_acyclic(compiled)
        _validate_reachable(compiled)
        _validate_edge_interfaces(compiled)
        return compiled


def validate_graph(graph: CompiledGraph | Module) -> CompiledGraph:
    """Validate a graph module or compiled graph."""

    return GraphValidator().validate(graph)


def _validate_references(graph: CompiledGraph) -> None:
    if not graph.entry_nodes:
        raise AgentNetValidationError("Graph requires at least one entry node")
    if not graph.output_nodes:
        raise AgentNetValidationError("Graph requires at least one output node")

    for node_name in (*graph.entry_nodes, *graph.output_nodes):
        if node_name not in graph.nodes:
            raise AgentNetValidationError(f"Graph references unknown node {node_name!r}")

    for source, targets in graph.edges.items():
        if source not in graph.nodes:
            raise AgentNetValidationError(f"Graph edge source {source!r} is not a node")
        for target in targets:
            if target not in graph.nodes:
                raise AgentNetValidationError(f"Graph edge target {target!r} is not a node")


def _validate_acyclic(graph: CompiledGraph) -> None:
    incoming = {node_name: 0 for node_name in graph.nodes}
    for targets in graph.edges.values():
        for target in targets:
            incoming[target] += 1

    ready = [node_name for node_name in graph.nodes if incoming[node_name] == 0]
    visited: list[str] = []
    while ready:
        node_name = ready.pop(0)
        visited.append(node_name)
        for target in graph.edges.get(node_name, ()):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)

    if len(visited) != len(graph.nodes):
        raise AgentNetValidationError("Graph contains a cycle")


def _validate_reachable(graph: CompiledGraph) -> None:
    reachable: set[str] = set()
    stack = list(graph.entry_nodes)
    while stack:
        node_name = stack.pop()
        if node_name in reachable:
            continue
        reachable.add(node_name)
        stack.extend(graph.edges.get(node_name, ()))

    unreachable = tuple(node_name for node_name in graph.nodes if node_name not in reachable)
    if unreachable:
        raise AgentNetValidationError(f"Graph has unreachable nodes: {', '.join(unreachable)}")


def _validate_edge_interfaces(graph: CompiledGraph) -> None:
    for source_name, target_names in graph.edges.items():
        source_interface = _output_interface(graph.nodes[source_name])
        for target_name in target_names:
            target_interface = _input_interface(graph.nodes[target_name])
            if source_interface is None or target_interface is None:
                continue
            try:
                validate_interface_compatibility(source_interface, target_interface)
            except AgentNetValidationError as exc:
                raise AgentNetValidationError(
                    f"Graph edge {source_name!r} -> {target_name!r} "
                    f"has incompatible interfaces: {exc}"
                ) from exc


def _output_interface(module: Module) -> Interface | None:
    value = getattr(module, "interface", None)
    return value if isinstance(value, Interface) else None


def _input_interface(module: Module) -> Interface | None:
    value = getattr(module, "input_interface", None)
    return value if isinstance(value, Interface) else None
