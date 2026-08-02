import agentnet as an
from agentnet.constraints import Constraint, GraphConstraint


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


class MaxNodesConstraint(Constraint):
    def __init__(self, max_nodes: int) -> None:
        super().__init__(f"max_nodes_{max_nodes}")
        self.max_nodes = max_nodes

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, an.CompiledGraph) and len(candidate.nodes) <= self.max_nodes


def test_graph_constraint_evaluates_inner_constraint_on_compiled_graph() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    constraint = GraphConstraint(MaxNodesConstraint(2))

    result = constraint.evaluate(graph)

    assert result.passed is True
    assert result.metadata["result"]["constraint"] == "max_nodes_2"


def test_graph_constraint_fails_when_inner_constraint_fails() -> None:
    graph = an.Sequential(
        NamedModule("first"),
        NamedModule("second"),
        NamedModule("third"),
    )
    constraint = GraphConstraint(MaxNodesConstraint(2))

    result = constraint.evaluate(graph)

    assert result.passed is False
    assert result.blocks_candidate is True
    assert result.metadata["result"]["passed"] is False


def test_graph_constraint_accepts_compiled_graph() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    compiled = an.compile_graph(graph)
    constraint = GraphConstraint(MaxNodesConstraint(2))

    assert constraint.evaluate(compiled).passed is True


def test_graph_constraint_is_exported_from_package_root() -> None:
    assert an.GraphConstraint is GraphConstraint
