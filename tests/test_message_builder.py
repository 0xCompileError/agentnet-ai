from agentnet.agents.messages import MessageBuilder
from agentnet.core.state import AgentState


def test_message_builder_builds_system_history_and_user_messages() -> None:
    state = AgentState(name="planner")
    state.add_message("assistant", "Prior summary.")

    messages = MessageBuilder(instructions="Plan clearly.").build(
        "What next?",
        state=state,
    )

    assert messages == [
        {"content": "Plan clearly.", "role": "system"},
        {"content": "Prior summary.", "role": "assistant"},
        {"content": "What next?", "role": "user"},
    ]


def test_message_builder_omits_empty_instructions() -> None:
    messages = MessageBuilder().build("Hello")

    assert messages == [{"content": "Hello", "role": "user"}]
