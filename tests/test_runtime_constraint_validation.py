from typing import Any

import pytest

import agentnet as an
from agentnet.constraints import Constraint


class ReturningContextModule(an.Module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls = 0

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        self.calls += 1
        return context


class RecordingModule(an.Module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls = 0

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        self.calls += 1
        return f"{input}:{self.name}"


class NamedModuleConstraint(Constraint):
    def __init__(
        self,
        expected_name: str,
        *,
        kind: an.ConstraintKind = an.ConstraintKind.HARD,
    ) -> None:
        super().__init__(f"module_is_{expected_name}", kind=kind)
        self.expected_name = expected_name

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, an.Module) and candidate.name == self.expected_name


def test_run_validates_runtime_constraints_before_execution() -> None:
    module = ReturningContextModule("worker")
    context = an.RunContext(run_id="run-constraints")

    result = an.run(
        module,
        "input",
        context=context,
        constraints=[NamedModuleConstraint("worker")],
    )

    assert result is context
    assert module.calls == 1
    assert context.metadata["constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "module_is_worker",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_run_rejects_failed_hard_runtime_constraint_before_execution() -> None:
    module = ReturningContextModule("worker")
    context = an.RunContext(run_id="run-constraints")

    with pytest.raises(an.AgentNetValidationError, match="module_is_other"):
        an.run(
            module,
            "input",
            context=context,
            constraints=[NamedModuleConstraint("other")],
        )

    assert module.calls == 0
    assert context.metadata["constraint_results"][0]["blocks_candidate"] is True


def test_run_allows_failed_soft_runtime_constraint() -> None:
    module = ReturningContextModule("worker")
    context = an.RunContext(run_id="run-constraints")

    result = an.run(
        module,
        "input",
        context=context,
        constraints=[
            NamedModuleConstraint("other", kind=an.ConstraintKind.SOFT),
        ],
    )

    assert result is context
    assert module.calls == 1
    assert context.metadata["constraint_results"][0]["blocks_candidate"] is False
    assert context.metadata["constraint_results"][0]["passed"] is False


def test_run_integrates_scoped_graph_constraints_before_graph_execution() -> None:
    planner = RecordingModule("planner")
    writer = RecordingModule("writer")
    graph = an.Sequential(planner, writer)
    context = an.RunContext(run_id="run-graph-constraints")

    def is_writer_node(candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, an.Module) and candidate.name == "writer"

    result = an.run(
        graph,
        "draft",
        context=context,
        constraints=[
            an.GraphConstraint(
                an.TopologyConstraint(
                    allowed_modules=["RecordingModule"],
                    max_depth=2,
                    max_nodes=2,
                )
            ),
            an.NodeConstraint(
                "writer",
                an.CustomConstraint("writer_node", is_writer_node),
            ),
        ],
    )

    assert result == "draft:planner:writer"
    assert planner.calls == 1
    assert writer.calls == 1
    assert context.metadata["constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "graph:topology",
            "kind": "hard",
            "message": None,
            "passed": True,
        },
        {
            "blocks_candidate": False,
            "constraint": "node(writer):writer_node",
            "kind": "hard",
            "message": None,
            "passed": True,
        },
    ]


@pytest.mark.anyio
async def test_arun_validates_runtime_constraints() -> None:
    module = ReturningContextModule("worker")

    result = await an.arun(
        module,
        "input",
        constraints=[NamedModuleConstraint("worker")],
    )

    assert isinstance(result, an.RunContext)


def test_validate_runtime_constraints_is_exported_from_package_root() -> None:
    assert an.validate_runtime_constraints(
        ReturningContextModule("worker"),
        [NamedModuleConstraint("worker")],
    )[0].passed is True
