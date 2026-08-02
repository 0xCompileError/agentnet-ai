import agentnet as an
from agentnet.constraints import TopologyConstraint


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


def test_topology_constraint_limits_node_count() -> None:
    constraint = TopologyConstraint(max_nodes=2)

    assert constraint.evaluate(an.Sequential(NamedModule("a"), NamedModule("b"))).passed is True
    assert (
        constraint.evaluate(
            an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
        ).passed
        is False
    )


def test_topology_constraint_limits_branch_count() -> None:
    constraint = TopologyConstraint(max_branches=1)
    router = an.Router(
        router=NamedModule("router"),
        routes={"a": NamedModule("a"), "b": NamedModule("b")},
    )

    assert constraint.evaluate(router).passed is False


def test_topology_constraint_accepts_compiled_graph() -> None:
    graph = an.compile_graph(an.Sequential(NamedModule("a"), NamedModule("b")))

    assert TopologyConstraint(max_depth=2).evaluate(graph).passed is True
    assert TopologyConstraint(max_depth=1).evaluate(graph).passed is False


def test_topology_constraint_is_exported_from_package_root() -> None:
    assert an.TopologyConstraint is TopologyConstraint
