import pytest

import agentnet as an


class RecordModule(an.Module):
    def __init__(self, name: str, suffix: str) -> None:
        super().__init__(name)
        self.suffix = suffix
        self.inputs: list[object] = []
        self.contexts: list[an.RunContext | None] = []

    async def arun(self, input: object, context: object | None = None) -> str:
        self.inputs.append(input)
        self.contexts.append(context if isinstance(context, an.RunContext) else None)
        if isinstance(input, tuple):
            base = "|".join(str(item) for item in input)
        else:
            base = str(input)
        return f"{base}{self.suffix}"


@pytest.mark.anyio
async def test_dag_executes_dependencies_and_returns_single_leaf_output() -> None:
    start = RecordModule("start", "-start")
    left = RecordModule("left", "-left")
    right = RecordModule("right", "-right")
    merge = RecordModule("merge", "-merge")
    context = an.RunContext(run_id="run-1")
    dag = an.DAG(
        nodes={"start": start, "left": left, "right": right, "merge": merge},
        edges={
            "start": ("left", "right"),
            "left": ("merge",),
            "right": ("merge",),
        },
    )

    result = await dag.arun("input", context)

    assert result == "input-start-left|input-start-right-merge"
    assert start.inputs == ["input"]
    assert left.inputs == ["input-start"]
    assert right.inputs == ["input-start"]
    assert merge.inputs == [("input-start-left", "input-start-right")]
    assert merge.contexts == [context]


@pytest.mark.anyio
async def test_dag_returns_mapping_for_multiple_leaf_outputs() -> None:
    left = RecordModule("left", "-left")
    right = RecordModule("right", "-right")
    dag = an.DAG(nodes={"left": left, "right": right}, edges={})

    result = await dag.arun("input")

    assert result == {"left": "input-left", "right": "input-right"}


@pytest.mark.anyio
async def test_dag_rejects_cycles_during_execution() -> None:
    dag = an.DAG(
        nodes={"a": RecordModule("a", "-a"), "b": RecordModule("b", "-b")},
        edges={"a": ("b",), "b": ("a",)},
    )

    with pytest.raises(an.AgentNetExecutionError, match="cycle"):
        await dag.arun("input")


def test_dag_is_exported_from_package_root() -> None:
    from agentnet.graphs import DAG

    assert an.DAG is DAG
