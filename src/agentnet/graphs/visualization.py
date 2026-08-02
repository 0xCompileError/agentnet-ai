"""Graph visualization helpers."""

import re

from agentnet.core import AgentNetConfigurationError, Module
from agentnet.graphs.compiler import CompiledGraph
from agentnet.graphs.validator import validate_graph


class GraphVisualizer:
    """Render graph structures into text diagrams."""

    def render(self, graph: CompiledGraph | Module, *, format: str = "mermaid") -> str:
        if format != "mermaid":
            raise AgentNetConfigurationError(f"Unsupported graph visualization format {format!r}")
        return self.to_mermaid(graph)

    def to_mermaid(self, graph: CompiledGraph | Module) -> str:
        compiled = validate_graph(graph)
        lines = ["graph TD"]
        for node_name in compiled.nodes:
            lines.append(f'    {_node_id(node_name)}["{_label(node_name)}"]')
        for source, targets in compiled.edges.items():
            for target in targets:
                lines.append(f"    {_node_id(source)} --> {_node_id(target)}")
        return "\n".join(lines)


def visualize_graph(graph: CompiledGraph | Module, *, format: str = "mermaid") -> str:
    """Render a graph module or compiled graph as text."""

    return GraphVisualizer().render(graph, format=format)


def _node_id(name: str) -> str:
    node_id = re.sub(r"\W+", "_", name).strip("_")
    if not node_id:
        node_id = "node"
    if node_id[0].isdigit():
        node_id = f"node_{node_id}"
    return node_id


def _label(name: str) -> str:
    return name.replace('"', '\\"')
