"""Structural graph compiler."""

from __future__ import annotations

from dataclasses import dataclass

from agentnet.core import AgentNetConfigurationError, Module
from agentnet.graphs.dag import DAG
from agentnet.graphs.parallel import Parallel
from agentnet.graphs.reducer import Reducer
from agentnet.graphs.router import Router
from agentnet.graphs.sequential import Sequential


@dataclass(frozen=True, slots=True, init=False)
class CompiledGraph:
    """Compiled graph structure used by validation and visualization."""

    nodes: dict[str, Module]
    edges: dict[str, tuple[str, ...]]
    entry_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]

    def __init__(
        self,
        *,
        nodes: dict[str, Module],
        edges: dict[str, tuple[str, ...]],
        entry_nodes: tuple[str, ...],
        output_nodes: tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "nodes", dict(nodes))
        object.__setattr__(
            self,
            "edges",
            {node_name: tuple(edges.get(node_name, ())) for node_name in nodes},
        )
        object.__setattr__(self, "entry_nodes", tuple(entry_nodes))
        object.__setattr__(self, "output_nodes", tuple(output_nodes))


class GraphCompiler:
    """Compile graph modules into explicit node and edge structures."""

    def compile(self, module: Module) -> CompiledGraph:
        if isinstance(module, Sequential):
            return self._compile_sequential(module)
        if isinstance(module, Parallel):
            return self._compile_parallel(module)
        if isinstance(module, Router):
            return self._compile_router(module)
        if isinstance(module, Reducer):
            return self._compile_reducer(module)
        if isinstance(module, DAG):
            return self._compile_dag(module)
        return CompiledGraph(
            nodes={module.name: module},
            edges={module.name: ()},
            entry_nodes=(module.name,),
            output_nodes=(module.name,),
        )

    def _compile_sequential(self, graph: Sequential) -> CompiledGraph:
        nodes = _nodes_by_name(graph.modules)
        module_names = tuple(nodes)
        edges = {
            node_name: (module_names[index + 1],) if index + 1 < len(module_names) else ()
            for index, node_name in enumerate(module_names)
        }
        return CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_nodes=(module_names[0],),
            output_nodes=(module_names[-1],),
        )

    def _compile_parallel(self, graph: Parallel) -> CompiledGraph:
        nodes = _nodes_by_name(graph.modules)
        entry_nodes = tuple(nodes)
        edges: dict[str, tuple[str, ...]] = {node_name: () for node_name in nodes}
        output_nodes = entry_nodes

        if graph.reducer is not None:
            _add_unique_node(nodes, graph.reducer)
            reducer_name = graph.reducer.name
            edges[reducer_name] = ()
            for node_name in entry_nodes:
                edges[node_name] = (reducer_name,)
            output_nodes = (reducer_name,)

        return CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_nodes=entry_nodes,
            output_nodes=output_nodes,
        )

    def _compile_router(self, graph: Router) -> CompiledGraph:
        nodes = {graph.router.name: graph.router}
        route_modules = tuple(graph.routes.values())
        for module in route_modules:
            _add_unique_node(nodes, module)
        if graph.fallback is not None:
            _add_unique_node(nodes, graph.fallback)

        output_nodes = tuple(module.name for module in route_modules)
        if graph.fallback is not None and graph.fallback.name not in output_nodes:
            output_nodes = (*output_nodes, graph.fallback.name)
        edges: dict[str, tuple[str, ...]] = {node_name: () for node_name in nodes}
        edges[graph.router.name] = output_nodes
        return CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_nodes=(graph.router.name,),
            output_nodes=output_nodes,
        )

    def _compile_reducer(self, graph: Reducer) -> CompiledGraph:
        reducer_name = graph.reducer.name
        return CompiledGraph(
            nodes={reducer_name: graph.reducer},
            edges={reducer_name: ()},
            entry_nodes=(reducer_name,),
            output_nodes=(reducer_name,),
        )

    def _compile_dag(self, graph: DAG) -> CompiledGraph:
        entry_nodes = tuple(
            node_name for node_name in graph.nodes if not graph.predecessors[node_name]
        )
        return CompiledGraph(
            nodes=dict(graph.nodes),
            edges={node_name: tuple(graph.edges.get(node_name, ())) for node_name in graph.nodes},
            entry_nodes=entry_nodes,
            output_nodes=graph.output_nodes,
        )


def compile_graph(module: Module) -> CompiledGraph:
    """Compile a module into a structural graph representation."""

    return GraphCompiler().compile(module)


def _nodes_by_name(modules: tuple[Module, ...]) -> dict[str, Module]:
    nodes: dict[str, Module] = {}
    for module in modules:
        _add_unique_node(nodes, module)
    return nodes


def _add_unique_node(nodes: dict[str, Module], module: Module) -> None:
    if module.name in nodes:
        raise AgentNetConfigurationError(f"Duplicate graph node name {module.name!r}")
    nodes[module.name] = module
