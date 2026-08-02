import agentnet as an
from agentnet.llms.response import ChatResponse


def test_chat_response_normalizes_model_and_usage() -> None:
    response = ChatResponse(
        content="Hello",
        model="strong",
        usage={"input_tokens": 1, "output_tokens": 2},
        finish_reason="stop",
    )

    assert response.content == "Hello"
    assert response.model == an.ModelRef("strong")
    assert response.usage == {"input_tokens": 1, "output_tokens": 2}
    assert response.finish_reason == "stop"


def test_chat_response_round_trips_to_dict() -> None:
    response = ChatResponse(content="Hello", model=an.ModelRef("strong"))

    assert ChatResponse.from_dict(response.to_dict()) == response


def test_chat_response_is_exported_from_package_root() -> None:
    assert an.ChatResponse is ChatResponse
