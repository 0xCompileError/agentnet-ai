import agentnet as an


def test_chat_event_serializes_without_mutating_inputs() -> None:
    usage = {"completion_tokens": 1}
    metadata = {"index": 0}

    event = an.ChatEvent(
        delta="Hel",
        model="strong",
        content="Hello",
        finish_reason="stop",
        metadata=metadata,
        usage=usage,
    )
    usage["completion_tokens"] = 99
    metadata["index"] = 1

    assert event.to_dict() == {
        "content": "Hello",
        "delta": "Hel",
        "finish_reason": "stop",
        "metadata": {"index": 0},
        "model": {"alias": "strong", "model": None, "provider": None},
        "usage": {"completion_tokens": 1},
    }
    assert an.ChatEvent.from_dict(event.to_dict()) == event


def test_chat_event_can_be_created_from_chat_response() -> None:
    response = an.ChatResponse(
        content="Hello",
        finish_reason="stop",
        metadata={"provider": "fake"},
        model="strong",
        usage={"completion_tokens": 1},
    )

    assert an.ChatEvent.from_response(response) == an.ChatEvent(
        content="Hello",
        delta="Hello",
        finish_reason="stop",
        metadata={"provider": "fake"},
        model="strong",
        usage={"completion_tokens": 1},
    )
