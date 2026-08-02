import pytest

import agentnet as an


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


def _modules(count: int) -> list[NamedModule]:
    return [NamedModule(f"node_{index}") for index in range(count)]


def test_build_shape_creates_sequential_for_single_branch() -> None:
    modules = _modules(3)

    graph = an.build_shape((1, 3), modules, name="chain")

    assert isinstance(graph, an.Sequential)
    assert graph.name == "chain"
    assert graph.modules == tuple(modules)


def test_build_shape_creates_parallel_sequential_branches() -> None:
    modules = _modules(4)

    graph = an.build_shape((2, 2), modules, name="fanout")

    assert isinstance(graph, an.Parallel)
    assert graph.name == "fanout"
    assert len(graph.modules) == 2
    first_branch = graph.modules[0]
    second_branch = graph.modules[1]
    assert isinstance(first_branch, an.Sequential)
    assert isinstance(second_branch, an.Sequential)
    assert first_branch.modules == (modules[0], modules[1])
    assert second_branch.modules == (modules[2], modules[3])


def test_build_shape_validates_shape_and_module_count() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="positive"):
        an.build_shape((0, 2), _modules(2))

    with pytest.raises(an.AgentNetConfigurationError, match="expected 4"):
        an.build_shape((2, 2), _modules(3))


def test_build_shape_is_exported_from_package_root() -> None:
    from agentnet.graphs import build_shape

    assert an.build_shape is build_shape
