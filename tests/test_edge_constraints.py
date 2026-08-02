import agentnet as an
from agentnet.constraints import Constraint, EdgeConstraint, GraphEdge


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


class EdgeEndpointsConstraint(Constraint):
    def __init__(self, source: str, target: str) -> None:
        super().__init__(f"edge_{source}_to_{target}")
        self.source = source
        self.target = target

    def check(self, candidate: object, context: object | None = None) -> bool:
        return (
            isinstance(candidate, GraphEdge)
            and candidate.source == self.source
            and candidate.target == self.target
            and candidate.source_module.name == self.source
            and candidate.target_module.name == self.target
        )


def test_edge_constraint_evaluates_inner_constraint_on_graph_edge() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    constraint = EdgeConstraint(
        "first",
        "second",
        EdgeEndpointsConstraint("first", "second"),
    )

    result = constraint.evaluate(graph)

    assert result.passed is True
    assert result.metadata["edge"] == {"source": "first", "target": "second"}
    assert result.metadata["result"]["constraint"] == "edge_first_to_second"


def test_edge_constraint_fails_when_edge_is_missing() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    constraint = EdgeConstraint(
        "second",
        "first",
        EdgeEndpointsConstraint("second", "first"),
    )

    result = constraint.evaluate(graph)

    assert result.passed is False
    assert result.blocks_candidate is True
    assert result.metadata == {"edge": {"source": "second", "target": "first"}}


def test_edge_constraint_accepts_compiled_graph() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    compiled = an.compile_graph(graph)
    constraint = EdgeConstraint(
        "first",
        "second",
        EdgeEndpointsConstraint("first", "second"),
    )

    assert constraint.evaluate(compiled).passed is True


def test_edge_constraint_is_exported_from_package_root() -> None:
    assert an.EdgeConstraint is EdgeConstraint
    assert an.GraphEdge is GraphEdge
