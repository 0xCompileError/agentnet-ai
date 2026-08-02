import agentnet as an
from agentnet.constraints import Constraint, NodeConstraint


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


class ModuleNameConstraint(Constraint):
    def __init__(self, expected_name: str) -> None:
        super().__init__(f"module_name_{expected_name}")
        self.expected_name = expected_name

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, an.Module) and candidate.name == self.expected_name


def test_node_constraint_evaluates_inner_constraint_on_named_graph_node() -> None:
    first = NamedModule("first")
    second = NamedModule("second")
    graph = an.Sequential(first, second)
    constraint = NodeConstraint("second", ModuleNameConstraint("second"))

    result = constraint.evaluate(graph)

    assert result.passed is True
    assert result.metadata["node"] == "second"
    assert result.metadata["result"]["constraint"] == "module_name_second"


def test_node_constraint_fails_when_node_is_missing() -> None:
    graph = an.Sequential(NamedModule("first"), NamedModule("second"))
    constraint = NodeConstraint("missing", ModuleNameConstraint("missing"))

    result = constraint.evaluate(graph)

    assert result.passed is False
    assert result.blocks_candidate is True
    assert result.metadata == {"node": "missing"}


def test_node_constraint_accepts_compiled_graph_and_plain_module() -> None:
    module = NamedModule("single")
    compiled = an.compile_graph(module)
    constraint = NodeConstraint("single", ModuleNameConstraint("single"))

    assert constraint.evaluate(compiled).passed is True
    assert constraint.evaluate(module).passed is True


def test_node_constraint_is_exported_from_package_root() -> None:
    assert an.NodeConstraint is NodeConstraint
