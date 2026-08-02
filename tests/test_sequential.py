import pytest

import agentnet as an


class AppendModule(an.Module):
    def __init__(self, name: str, suffix: str) -> None:
        super().__init__(name)
        self.suffix = suffix
        self.contexts: list[an.RunContext | None] = []

    async def arun(self, input: object, context: object | None = None) -> str:
        self.contexts.append(context if isinstance(context, an.RunContext) else None)
        return f"{input}{self.suffix}"


def test_sequential_stores_modules_defensively() -> None:
    first = AppendModule("first", "-a")
    modules = [first]

    sequence = an.Sequential(*modules, name="pipeline")
    modules.append(AppendModule("second", "-b"))

    assert isinstance(sequence, an.Module)
    assert sequence.name == "pipeline"
    assert sequence.modules == (first,)


@pytest.mark.anyio
async def test_sequential_runs_modules_in_order_with_context() -> None:
    first = AppendModule("first", "-a")
    second = AppendModule("second", "-b")
    sequence = an.Sequential(first, second)
    context = an.RunContext(run_id="run-1")

    result = await sequence.arun("start", context)

    assert result == "start-a-b"
    assert first.contexts == [context]
    assert second.contexts == [context]


def test_sequential_requires_at_least_one_module() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="at least one"):
        an.Sequential()


def test_sequential_is_exported_from_package_root() -> None:
    from agentnet.graphs import Sequential

    assert an.Sequential is Sequential
