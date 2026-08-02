import pytest

import agentnet as an


class CollectModule(an.Module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.inputs: list[object] = []
        self.contexts: list[an.RunContext | None] = []

    async def arun(self, input: object, context: object | None = None) -> str:
        self.inputs.append(input)
        self.contexts.append(context if isinstance(context, an.RunContext) else None)
        items = input if isinstance(input, tuple) else (input,)
        return " + ".join(str(item) for item in items)


class BranchModule(an.Module):
    def __init__(self, name: str, output: str) -> None:
        super().__init__(name)
        self.output = output

    async def arun(self, input: object, context: object | None = None) -> str:
        return f"{input}:{self.output}"


def test_reducer_stores_reducer_module() -> None:
    collector = CollectModule("collector")

    reducer = an.Reducer(collector, name="merge")

    assert isinstance(reducer, an.Module)
    assert reducer.name == "merge"
    assert reducer.reducer is collector


@pytest.mark.anyio
async def test_reducer_passes_input_to_reducer_module_with_context() -> None:
    collector = CollectModule("collector")
    reducer = an.Reducer(collector)
    context = an.RunContext(run_id="run-1")

    result = await reducer.arun(("a", "b"), context)

    assert result == "a + b"
    assert collector.inputs == [("a", "b")]
    assert collector.contexts == [context]


@pytest.mark.anyio
async def test_parallel_uses_optional_reducer() -> None:
    collector = CollectModule("collector")
    parallel = an.Parallel(
        BranchModule("first", "a"),
        BranchModule("second", "b"),
        reducer=collector,
    )

    result = await parallel.arun("input")

    assert result == "input:a + input:b"
    assert collector.inputs == [("input:a", "input:b")]


def test_reducer_is_exported_from_package_root() -> None:
    from agentnet.graphs import Reducer

    assert an.Reducer is Reducer
