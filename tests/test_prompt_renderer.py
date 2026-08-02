from agentnet.agents.messages import MessageBuilder
from agentnet.agents.prompts import PromptRenderer
from agentnet.core.state import AgentState


def test_prompt_renderer_returns_plain_input_without_state() -> None:
    prompt = PromptRenderer().render("What next?")

    assert prompt == "What next?"


def test_prompt_renderer_includes_reasoning_and_actions() -> None:
    state = AgentState(name="researcher")
    state.add_reasoning("Check the docs first.", model="strong")
    state.add_action("search_docs", arguments={"query": "agentnet"})

    prompt = PromptRenderer().render("What next?", state=state)

    assert prompt == (
        "What next?\n\n"
        "Previous reasoning:\n"
        "- Check the docs first.\n\n"
        "Previous actions:\n"
        '- search_docs {"query": "agentnet"}'
    )


def test_message_builder_uses_prompt_renderer_for_user_content() -> None:
    state = AgentState(name="researcher")
    state.add_action("search_docs", arguments={"query": "agentnet"})

    messages = MessageBuilder().build("What next?", state=state)

    assert messages[-1] == {
        "content": 'What next?\n\nPrevious actions:\n- search_docs {"query": "agentnet"}',
        "role": "user",
    }
