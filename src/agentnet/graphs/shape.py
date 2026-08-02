"""Shape DSL helpers for graph construction."""

from collections.abc import Sequence

from agentnet.core import AgentNetConfigurationError, Module
from agentnet.graphs.parallel import Parallel
from agentnet.graphs.sequential import Sequential


def build_shape(
    shape: tuple[int, int],
    modules: Sequence[Module],
    *,
    name: str = "shape",
) -> Module:
    """Build a graph from a ``(branches, depth)`` shape."""

    branches, depth = shape
    if branches < 1 or depth < 1:
        raise AgentNetConfigurationError("Shape dimensions must be positive")

    expected = branches * depth
    if len(modules) != expected:
        raise AgentNetConfigurationError(
            f"Shape {shape!r} expected {expected} modules, received {len(modules)}"
        )

    module_tuple = tuple(modules)
    if branches == 1:
        return Sequential(*module_tuple, name=name)

    branch_graphs = [
        Sequential(
            *module_tuple[index * depth : (index + 1) * depth],
            name=f"{name}.branch_{index}",
        )
        for index in range(branches)
    ]
    return Parallel(*branch_graphs, name=name)
