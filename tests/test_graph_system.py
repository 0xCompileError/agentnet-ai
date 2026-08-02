import pytest

import agentnet as an


class AppendModule(an.Module):
    def __init__(self, name: str, suffix: str) -> None:
        super().__init__(name)
        self.suffix = suffix

    async def arun(self, input: object, context: object | None = None) -> str:
        return f"{input}{self.suffix}"


class JoinModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> str:
        items = input if isinstance(input, tuple) else (input,)
        return " + ".join(str(item) for item in items)


@pytest.mark.anyio
async def test_composed_graph_runs_through_runtime_entrypoint() -> None:
    graph = an.Sequential(
        an.Parallel(
            AppendModule("left", "-left"),
            AppendModule("right", "-right"),
            reducer=JoinModule("join"),
        ),
        AppendModule("tail", "-tail"),
    )

    result = await an.arun(graph, "input", an.RunContext(run_id="run-1"))

    assert result == "input-left + input-right-tail"
