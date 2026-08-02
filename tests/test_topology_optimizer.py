import pytest

import agentnet as an
from agentnet.constraints import Constraint, GraphConstraint


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


class MaxNodesConstraint(Constraint):
    def __init__(
        self,
        max_nodes: int,
        *,
        kind: an.ConstraintKind = an.ConstraintKind.HARD,
    ) -> None:
        super().__init__(f"max_nodes_{max_nodes}", kind=kind)
        self.max_nodes = max_nodes

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, an.CompiledGraph) and len(candidate.nodes) <= self.max_nodes


def test_topology_optimizer_skips_candidates_outside_search_space() -> None:
    small = an.Sequential(NamedModule("a"), NamedModule("b"))
    large = an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
    optimizer = an.TopologyOptimizer(
        search_space=an.TopologySearchSpace(max_nodes=2),
    )

    result = optimizer.optimize(
        [large, small],
        scorer=lambda graph: float(len(graph.nodes)),
    )

    assert result.module is small
    assert result.score == 2.0
    assert result.compiled_graph.nodes == {"a": small.modules[0], "b": small.modules[1]}
    assert result.metadata["rejected_candidates"] == 1


def test_topology_optimizer_applies_graph_constraints() -> None:
    small = an.Sequential(NamedModule("a"), NamedModule("b"))
    large = an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
    optimizer = an.TopologyOptimizer(
        constraints=[GraphConstraint(MaxNodesConstraint(2))],
    )

    result = optimizer.optimize(
        [large, small],
        scorer=lambda graph: float(len(graph.nodes)),
    )

    assert result.module is small
    assert result.constraint_results[0].passed is True


def test_topology_optimizer_allows_soft_constraint_violations_by_score() -> None:
    small = an.Sequential(NamedModule("a"), NamedModule("b"))
    large = an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
    optimizer = an.TopologyOptimizer(
        constraints=[GraphConstraint(MaxNodesConstraint(2, kind=an.ConstraintKind.SOFT))],
    )

    result = optimizer.optimize(
        [small, large],
        scorer=lambda graph: float(len(graph.nodes)),
    )

    assert result.module is large
    assert result.constraint_results[0].passed is False


def test_topology_optimizer_rejects_when_no_candidate_is_valid() -> None:
    large = an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
    optimizer = an.TopologyOptimizer(
        search_space=an.TopologySearchSpace(max_nodes=2),
    )

    with pytest.raises(an.AgentNetValidationError, match="No topology candidate"):
        optimizer.optimize([large], scorer=lambda graph: float(len(graph.nodes)))


def test_topology_optimizer_reports_final_candidate_counts() -> None:
    best = an.Sequential(NamedModule("a"), NamedModule("b"))
    rejected = an.Sequential(NamedModule("a"), NamedModule("b"), NamedModule("c"))
    accepted = an.Sequential(NamedModule("x"), NamedModule("y"))
    optimizer = an.TopologyOptimizer(
        constraints=[GraphConstraint(MaxNodesConstraint(2))],
        metadata={"optimizer": "topology"},
    )

    result = optimizer.optimize(
        [best, rejected, accepted],
        scorer=lambda graph: 10.0 if "a" in graph.nodes else 1.0,
    )

    assert result.module is best
    assert result.metadata["optimizer"] == "topology"
    assert result.metadata["evaluated_candidates"] == 2
    assert result.metadata["rejected_candidates"] == 1
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "graph:max_nodes_2",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_topology_optimizer_is_exported_from_package_root() -> None:
    assert an.TopologyOptimizer is not None
    assert an.TopologySearchSpace is not None
    assert an.TopologyOptimizationResult is not None
