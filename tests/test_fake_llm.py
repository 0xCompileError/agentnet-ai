import pytest

import agentnet as an
from agentnet.llms.fake import FakeLLM


@pytest.mark.anyio
async def test_fake_llm_returns_queued_chat_responses() -> None:
    llm = FakeLLM(responses=["first", "second"])
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
    )

    first = await llm.complete(request)
    second = await llm.complete(request)

    assert first == an.ChatResponse(content="first", model="strong")
    assert second == an.ChatResponse(content="second", model="strong")
    assert llm.requests == [request, request]


@pytest.mark.anyio
async def test_fake_llm_stream_yields_single_response_event() -> None:
    llm = FakeLLM(responses=["streamed"])
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
    )

    events = [event async for event in llm.stream(request)]

    assert events == [
        an.ChatEvent(content="streamed", delta="streamed", model="strong")
    ]


def test_fake_llm_satisfies_backend_protocol() -> None:
    assert isinstance(FakeLLM(), an.LLMBackend)
    assert an.FakeLLM is FakeLLM
