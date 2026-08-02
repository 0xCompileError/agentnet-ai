import pytest

import agentnet as an
import agentnet.agents.react as react_module


class FlakyTransportLLM:
    def __init__(self, failures_before_success: int, response: str = "final answer") -> None:
        self.name = "flaky"
        self.model = "flaky-model"
        self.failures_before_success = failures_before_success
        self.requests: list[an.ChatRequest] = []
        self.response = response

    async def complete(self, request: an.ChatRequest) -> an.ChatResponse:
        self.requests.append(request)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise TimeoutError("gateway timed out")
        return an.ChatResponse(content=self.response, model=request.model)


class QueuedContentResponse:
    def __init__(self, content: object, model: object) -> None:
        self.content = content
        self.model = model


class QueuedContentLLM:
    def __init__(self, responses: list[object], name: str = "queued") -> None:
        self.name = name
        self.model = f"{name}-model"
        self.requests: list[an.ChatRequest] = []
        self.responses = list(responses)

    async def complete(self, request: an.ChatRequest) -> QueuedContentResponse:
        self.requests.append(request)
        content = self.responses.pop(0) if self.responses else ""
        return QueuedContentResponse(content=content, model=request.model)


def test_react_agent_stores_configuration_defensively() -> None:
    metadata = {"team": "ops"}
    tools = ["search_docs"]

    agent = an.ReActAgent(
        "planner",
        instructions="Plan the work.",
        llms=[an.ModelRef("strong"), "cheap"],
        max_steps=4,
        metadata=metadata,
        tools=tools,
    )
    metadata["team"] = "changed"
    tools.append("query_metrics")

    assert isinstance(agent, an.Module)
    assert agent.name == "planner"
    assert agent.instructions == "Plan the work."
    assert agent.llms == (an.ModelRef("strong"), an.ModelRef("cheap"))
    assert agent.max_steps == 4
    assert agent.metadata == {"team": "ops"}
    assert agent.tools == ("search_docs",)


def test_react_agent_exposes_per_agent_tool_whitelist() -> None:
    agent = an.ReActAgent("researcher", tools=["search_docs", "query_metrics"])

    assert agent.allowed_tools == ("search_docs", "query_metrics")
    assert agent.allows_tool("search_docs") is True
    assert agent.allows_tool("create_ticket") is False


def test_react_agent_enforces_tool_permissions() -> None:
    agent = an.ReActAgent("researcher", tools=["search_docs"])

    assert agent.require_tool("search_docs") == "search_docs"

    with pytest.raises(an.AgentNetExecutionError, match="not allowed"):
        agent.require_tool("create_ticket")


def test_react_agent_state_round_trips_without_live_backend_secrets() -> None:
    llm = an.LiteLLM(
        base_url="https://llm-gateway.example",
        model="gpt-4o",
        name="gateway",
        token="secret-token",
    )
    agent = an.ReActAgent(
        "writer",
        instructions="Write clearly.",
        llms=[llm],
        max_steps=3,
        metadata={"purpose": "drafting"},
        tools=["search_docs"],
    )

    state = agent.state_dict()

    assert "secret-token" not in str(state)
    assert "llm-gateway.example" not in str(state)
    assert state == {
        "instructions": "Write clearly.",
        "llms": [{"alias": "gateway", "model": "gpt-4o", "provider": "LiteLLM"}],
        "max_steps": 3,
        "metadata": {"purpose": "drafting"},
        "name": "writer",
        "tools": ["search_docs"],
    }

    restored = an.ReActAgent("empty")
    restored.load_state_dict(state)

    assert restored.name == "writer"
    assert restored.instructions == "Write clearly."
    assert restored.llms == (an.ModelRef("gateway", provider="LiteLLM", model="gpt-4o"),)
    assert restored.max_steps == 3
    assert restored.metadata == {"purpose": "drafting"}
    assert restored.tools == ("search_docs",)


@pytest.mark.anyio
async def test_react_agent_execution_loop_calls_first_live_llm() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", instructions="Plan clearly.", llms=[llm])

    result = await agent.arun("What should we do?")

    assert result == "final answer"
    assert llm.requests == [
        an.ChatRequest(
            model="fake",
            messages=[
                {"content": "Plan clearly.", "role": "system"},
                {"content": "What should we do?", "role": "user"},
            ],
        )
    ]


@pytest.mark.anyio
async def test_react_agent_retries_transport_failures_before_success() -> None:
    llm = FlakyTransportLLM(failures_before_success=2)
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        retry_policy=an.RetryPolicy(transport_retries=2),
    )

    result = await agent.arun("input")

    assert result == "final answer"
    assert len(llm.requests) == 3


@pytest.mark.anyio
async def test_react_agent_raises_after_transport_retries_are_exhausted() -> None:
    llm = FlakyTransportLLM(failures_before_success=2)
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        retry_policy=an.RetryPolicy(transport_retries=1),
    )

    with pytest.raises(an.AgentNetExecutionError, match="transport failure"):
        await agent.arun("input")

    assert len(llm.requests) == 2


@pytest.mark.anyio
async def test_react_agent_transport_retries_respect_total_attempt_cap() -> None:
    llm = FlakyTransportLLM(failures_before_success=2)
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        retry_policy=an.RetryPolicy(transport_retries=3, max_total_attempts=2),
    )

    with pytest.raises(an.AgentNetExecutionError, match="transport failure"):
        await agent.arun("input")

    assert len(llm.requests) == 2


@pytest.mark.anyio
async def test_react_agent_waits_with_backoff_between_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(react_module.anyio, "sleep", capture_sleep)
    llm = FlakyTransportLLM(failures_before_success=2)
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        retry_policy=an.RetryPolicy(
            transport_retries=2,
            backoff="exponential",
            backoff_base_seconds=0.25,
        ),
    )

    result = await agent.arun("input")

    assert result == "final answer"
    assert delays == [0.25, 0.5]


@pytest.mark.anyio
async def test_react_agent_records_transport_retry_events() -> None:
    llm = FlakyTransportLLM(failures_before_success=1)
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        retry_policy=an.RetryPolicy(transport_retries=1),
    )
    context = an.RunContext(run_id="run-1")

    await agent.arun("input", context)

    assert context.metadata["retry_events"] == [
        {
            "agent": "planner",
            "attempt": 1,
            "delay_seconds": 0.0,
            "error_type": "TimeoutError",
            "model": "flaky",
            "next_attempt": 2,
            "reason": "transport",
            "type": "retry.started",
        }
    ]
    assert context.metadata["retry_metrics"] == {
        "quality_retries": 0,
        "total_backoff_seconds": 0.0,
        "total_retries": 1,
        "transport_retries": 1,
    }


@pytest.mark.anyio
async def test_react_agent_switches_models_after_transport_failure() -> None:
    primary = FlakyTransportLLM(failures_before_success=1)
    backup = an.FakeLLM(responses=["backup answer"], name="backup")
    agent = an.ReActAgent(
        "planner",
        llms=[primary, backup],
        retry_policy=an.RetryPolicy(transport_retries=0, fallback_on=["timeout"]),
    )

    result = await agent.arun("input")

    assert result == "backup answer"
    assert len(primary.requests) == 1
    assert backup.requests == [
        an.ChatRequest(
            model="backup",
            messages=[{"content": "input", "role": "user"}],
        )
    ]


@pytest.mark.anyio
async def test_react_agent_does_not_switch_models_for_disallowed_failure() -> None:
    primary = FlakyTransportLLM(failures_before_success=1)
    backup = an.FakeLLM(responses=["backup answer"], name="backup")
    agent = an.ReActAgent(
        "planner",
        llms=[primary, backup],
        retry_policy=an.RetryPolicy(transport_retries=0, fallback_on=["schema_failure"]),
    )

    with pytest.raises(an.AgentNetExecutionError, match="transport failure"):
        await agent.arun("input")

    assert len(primary.requests) == 1
    assert backup.requests == []


@pytest.mark.anyio
async def test_react_agent_retries_quality_failures_before_success() -> None:
    llm = QueuedContentLLM(responses=["not a mapping", {"summary": "valid"}])
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(quality_retries=1, transport_retries=0),
    )

    result = await agent.arun("input")

    assert result == {"summary": "valid"}
    assert len(llm.requests) == 2


@pytest.mark.anyio
async def test_react_agent_waits_with_backoff_between_quality_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(react_module.anyio, "sleep", capture_sleep)
    llm = QueuedContentLLM(responses=["not a mapping", {"summary": "valid"}])
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(
            quality_retries=1,
            transport_retries=0,
            backoff="constant",
            backoff_base_seconds=0.2,
        ),
    )

    result = await agent.arun("input")

    assert result == {"summary": "valid"}
    assert delays == [0.2]


@pytest.mark.anyio
async def test_react_agent_records_quality_retry_events() -> None:
    llm = QueuedContentLLM(responses=["not a mapping", {"summary": "valid"}])
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(quality_retries=1, transport_retries=0),
    )
    context = an.RunContext(run_id="run-1")

    await agent.arun("input", context)

    assert context.metadata["retry_events"] == [
        {
            "agent": "planner",
            "attempt": 1,
            "delay_seconds": 0.0,
            "error_type": "AgentNetValidationError",
            "model": "queued",
            "next_attempt": 2,
            "reason": "quality",
            "type": "retry.started",
        }
    ]
    assert context.metadata["retry_metrics"] == {
        "quality_retries": 1,
        "total_backoff_seconds": 0.0,
        "total_retries": 1,
        "transport_retries": 0,
    }


@pytest.mark.anyio
async def test_react_agent_switches_models_after_schema_failure() -> None:
    primary = QueuedContentLLM(responses=["not a mapping"], name="primary")
    backup = QueuedContentLLM(responses=[{"summary": "valid"}], name="backup")
    agent = an.ReActAgent(
        "planner",
        llms=[primary, backup],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(
            quality_retries=0,
            transport_retries=0,
            fallback_on=["schema_failure"],
        ),
    )

    result = await agent.arun("input")

    assert result == {"summary": "valid"}
    assert len(primary.requests) == 1
    assert len(backup.requests) == 1


@pytest.mark.anyio
async def test_react_agent_raises_after_quality_retries_are_exhausted() -> None:
    llm = QueuedContentLLM(responses=["bad", "still bad"])
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(quality_retries=1, transport_retries=0),
    )

    with pytest.raises(an.AgentNetValidationError, match="output"):
        await agent.arun("input")

    assert len(llm.requests) == 2


@pytest.mark.anyio
async def test_react_agent_quality_retries_respect_total_attempt_cap() -> None:
    llm = QueuedContentLLM(responses=["bad", {"summary": "valid"}])
    agent = an.ReActAgent(
        "planner",
        llms=[llm],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(quality_retries=3, transport_retries=0, max_total_attempts=1),
    )

    with pytest.raises(an.AgentNetValidationError, match="output"):
        await agent.arun("input")

    assert len(llm.requests) == 1


@pytest.mark.anyio
async def test_react_agent_records_reasoning_state_in_context() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", llms=[llm])
    context = an.RunContext(run_id="run-1")

    result = await agent.arun("What should we do?", context)

    agent_state = context.graph_state.get_agent_state("planner")
    assert result == "final answer"
    assert agent_state.reasoning == [
        {"content": "final answer", "model": "fake", "step": 0}
    ]
    assert agent_state.step == 1


@pytest.mark.anyio
async def test_react_agent_message_builder_includes_existing_state_messages() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", instructions="Plan clearly.", llms=[llm])
    context = an.RunContext(run_id="run-1")
    agent_state = an.AgentState(name="planner")
    agent_state.add_message("assistant", "Prior summary.")
    context.graph_state.set_agent_state(agent_state)

    await agent.arun("What next?", context)

    assert llm.requests == [
        an.ChatRequest(
            model="fake",
            messages=[
                {"content": "Plan clearly.", "role": "system"},
                {"content": "Prior summary.", "role": "assistant"},
                {"content": "What next?", "role": "user"},
            ],
        )
    ]


@pytest.mark.anyio
async def test_react_agent_validates_input_before_llm_call() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", input_schema=dict[str, str], llms=[llm])

    with pytest.raises(an.AgentNetValidationError, match="input"):
        await agent.arun("not a mapping")

    assert llm.requests == []


@pytest.mark.anyio
async def test_react_agent_validates_output_after_llm_call() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", output_schema=dict[str, str], llms=[llm])

    with pytest.raises(an.AgentNetValidationError, match="output"):
        await agent.arun("input")

    assert len(llm.requests) == 1


@pytest.mark.anyio
async def test_react_agent_validates_output_with_interface() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent(
        "planner",
        interface=an.Interface(schema=dict[str, str]),
        llms=[llm],
    )

    with pytest.raises(an.AgentNetValidationError, match="output"):
        await agent.arun("input")

    assert len(llm.requests) == 1


@pytest.mark.anyio
async def test_react_agent_blocks_execution_after_max_steps() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", llms=[llm], max_steps=1)
    context = an.RunContext(run_id="run-1")
    context.graph_state.set_agent_state(an.AgentState(name="planner", step=1))

    with pytest.raises(an.AgentNetExecutionError, match="max steps"):
        await agent.arun("input", context)

    assert llm.requests == []


@pytest.mark.anyio
async def test_react_agent_allows_execution_before_max_steps() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", llms=[llm], max_steps=1)
    context = an.RunContext(run_id="run-1")

    result = await agent.arun("input", context)

    assert result == "final answer"
    assert context.graph_state.get_agent_state("planner").step == 1


@pytest.mark.anyio
async def test_react_agent_runtime_entrypoint_records_context_state() -> None:
    llm = an.FakeLLM(responses=["final answer"])
    agent = an.ReActAgent("planner", llms=[llm])
    context = an.RunContext(run_id="run-1")

    result = await an.arun(agent, "input", context)

    agent_state = context.graph_state.get_agent_state("planner")
    assert result == "final answer"
    assert agent_state.reasoning == [
        {"content": "final answer", "model": "fake", "step": 0}
    ]
    assert agent_state.step == 1


@pytest.mark.anyio
async def test_react_agent_execution_loop_requires_live_llm_backend() -> None:
    agent = an.ReActAgent("planner", llms=[an.ModelRef("strong")])

    with pytest.raises(an.AgentNetConfigurationError):
        await agent.arun("input")
