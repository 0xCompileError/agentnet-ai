import pytest

import agentnet as an
from agentnet.llms.request import ChatRequest


def test_chat_request_normalizes_model_ref_and_messages() -> None:
    request = ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    assert request.model == an.ModelRef("strong")
    assert request.messages == ({"content": "Hello", "role": "user"},)
    assert request.metadata == {"temperature": 0}


def test_chat_request_round_trips_to_dict() -> None:
    request = ChatRequest(
        model=an.ModelRef("strong", provider="openai", model="gpt-4o"),
        messages=[{"content": "Hello", "role": "user"}],
    )

    assert ChatRequest.from_dict(request.to_dict()) == request


def test_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(an.AgentNetValidationError):
        ChatRequest(model="strong", messages=[])


def test_chat_request_is_exported_from_package_root() -> None:
    assert an.ChatRequest is ChatRequest
