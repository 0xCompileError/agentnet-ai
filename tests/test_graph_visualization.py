import pytest

import agentnet as an


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


def test_visualize_graph_renders_mermaid_edges() -> None:
    first = NamedModule("first")
    second = NamedModule("second")

    diagram = an.visualize_graph(an.Sequential(first, second))

    assert diagram == '\n'.join(
        [
            "graph TD",
            '    first["first"]',
            '    second["second"]',
            "    first --> second",
        ]
    )


def test_visualize_graph_renders_isolated_nodes() -> None:
    solo = NamedModule("solo")

    diagram = an.visualize_graph(solo)

    assert diagram == '\n'.join(["graph TD", '    solo["solo"]'])


def test_visualize_graph_rejects_unknown_format() -> None:
    solo = NamedModule("solo")

    with pytest.raises(an.AgentNetConfigurationError, match="format"):
        an.visualize_graph(solo, format="dot")


def test_graph_visualizer_is_exported_from_package_root() -> None:
    from agentnet.graphs import GraphVisualizer, visualize_graph

    assert an.GraphVisualizer is GraphVisualizer
    assert an.visualize_graph is visualize_graph
