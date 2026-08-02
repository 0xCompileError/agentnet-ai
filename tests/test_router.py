import pytest

import agentnet as an


class StaticModule(an.Module):
    def __init__(self, name: str, output: str) -> None:
        super().__init__(name)
        self.output = output
        self.inputs: list[object] = []
        self.contexts: list[an.RunContext | None] = []

    async def arun(self, input: object, context: object | None = None) -> str:
        self.inputs.append(input)
        self.contexts.append(context if isinstance(context, an.RunContext) else None)
        return self.output


def test_router_stores_routes_defensively() -> None:
    router_agent = StaticModule("router-agent", "billing")
    billing = StaticModule("billing", "billing result")
    routes = {"billing": billing}

    router = an.Router(router=router_agent, routes=routes, name="support-router")
    routes["technical"] = StaticModule("technical", "technical result")

    assert isinstance(router, an.Module)
    assert router.name == "support-router"
    assert router.router is router_agent
    assert router.routes == {"billing": billing}


@pytest.mark.anyio
async def test_router_runs_selected_route_with_original_input_and_context() -> None:
    router_agent = StaticModule("router-agent", " technical\n")
    billing = StaticModule("billing", "billing result")
    technical = StaticModule("technical", "technical result")
    router = an.Router(
        router=router_agent,
        routes={"billing": billing, "technical": technical},
    )
    context = an.RunContext(run_id="run-1")

    result = await router.arun("help me", context)

    assert result == "technical result"
    assert router_agent.inputs == ["help me"]
    assert technical.inputs == ["help me"]
    assert technical.contexts == [context]
    assert billing.inputs == []


@pytest.mark.anyio
async def test_router_uses_fallback_for_unknown_route() -> None:
    router_agent = StaticModule("router-agent", "unknown")
    fallback = StaticModule("general", "general result")
    router = an.Router(
        router=router_agent,
        routes={"billing": StaticModule("billing", "billing result")},
        fallback=fallback,
    )

    result = await router.arun("help me")

    assert result == "general result"
    assert fallback.inputs == ["help me"]


@pytest.mark.anyio
async def test_router_rejects_unknown_route_without_fallback() -> None:
    router_agent = StaticModule("router-agent", "unknown")
    router = an.Router(
        router=router_agent,
        routes={"billing": StaticModule("billing", "billing result")},
    )

    with pytest.raises(an.AgentNetExecutionError, match="unknown route"):
        await router.arun("help me")


def test_router_is_exported_from_package_root() -> None:
    from agentnet.graphs import Router

    assert an.Router is Router
