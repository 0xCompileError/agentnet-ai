import pytest

import agentnet as an


class BranchModule(an.Module):
    def __init__(self, name: str, suffix: str) -> None:
        super().__init__(name)
        self.suffix = suffix
        self.contexts: list[an.RunContext | None] = []

    async def arun(self, input: object, context: object | None = None) -> str:
        self.contexts.append(context if isinstance(context, an.RunContext) else None)
        return f"{input}{self.suffix}"


def test_parallel_stores_modules_defensively() -> None:
    first = BranchModule("first", "-a")
    modules = [first]

    parallel = an.Parallel(*modules, name="fanout")
    modules.append(BranchModule("second", "-b"))

    assert isinstance(parallel, an.Module)
    assert parallel.name == "fanout"
    assert parallel.modules == (first,)


@pytest.mark.anyio
async def test_parallel_runs_modules_with_same_input_and_context() -> None:
    first = BranchModule("first", "-a")
    second = BranchModule("second", "-b")
    parallel = an.Parallel(first, second)
    context = an.RunContext(run_id="run-1")

    result = await parallel.arun("start", context)

    assert result == ("start-a", "start-b")
    assert first.contexts == [context]
    assert second.contexts == [context]


def test_parallel_requires_at_least_one_module() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="at least one"):
        an.Parallel()


def test_parallel_is_exported_from_package_root() -> None:
    from agentnet.graphs import Parallel

    assert an.Parallel is Parallel
