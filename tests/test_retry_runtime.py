import pytest

import agentnet as an


class ScriptedResponse:
    def __init__(self, content: object, model: object) -> None:
        self.content = content
        self.model = model


class ScriptedLLM:
    def __init__(self, name: str, script: list[object]) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.requests: list[an.ChatRequest] = []
        self.script = list(script)

    async def complete(self, request: an.ChatRequest) -> ScriptedResponse:
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return ScriptedResponse(content=step, model=request.model)


@pytest.mark.anyio
async def test_retry_runtime_combines_transport_retry_and_schema_fallback() -> None:
    primary = ScriptedLLM("primary", [TimeoutError("timeout"), "not a mapping"])
    backup = ScriptedLLM("backup", [{"summary": "valid"}])
    context = an.RunContext(run_id="run-1")
    agent = an.ReActAgent(
        "planner",
        llms=[primary, backup],
        output_schema=an.Schema({"summary": str}),
        retry_policy=an.RetryPolicy(
            fallback_on=["schema_failure"],
            quality_retries=0,
            transport_retries=1,
        ),
    )

    result = await agent.arun("input", context)

    assert result == {"summary": "valid"}
    assert len(primary.requests) == 2
    assert len(backup.requests) == 1
    assert context.metadata["retry_events"] == [
        {
            "agent": "planner",
            "attempt": 1,
            "delay_seconds": 0.0,
            "error_type": "TimeoutError",
            "model": "primary",
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
