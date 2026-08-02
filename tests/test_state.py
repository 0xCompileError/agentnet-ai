import agentnet as an
from agentnet.core.state import AgentState, GraphState


def test_agent_state_records_messages_and_steps() -> None:
    state = AgentState(name="planner")

    state.add_message("user", "Plan the work.")
    state.advance()

    assert state.name == "planner"
    assert state.step == 1
    assert state.messages == [{"content": "Plan the work.", "role": "user"}]


def test_agent_state_records_reasoning_by_step() -> None:
    state = AgentState(name="planner")

    state.add_reasoning("Need to inspect the context.", model="strong")
    state.advance()
    state.add_reasoning("Now compare the options.")

    assert state.reasoning == [
        {"content": "Need to inspect the context.", "model": "strong", "step": 0},
        {"content": "Now compare the options.", "step": 1},
    ]


def test_agent_state_records_actions_by_step() -> None:
    state = AgentState(name="researcher")

    state.add_action(
        "search_docs",
        arguments={"query": "agentnet"},
        metadata={"approved": True},
    )
    state.advance()
    state.add_action("query_metrics")

    assert state.actions == [
        {
            "arguments": {"query": "agentnet"},
            "metadata": {"approved": True},
            "name": "search_docs",
            "step": 0,
        },
        {"name": "query_metrics", "step": 1},
    ]


def test_agent_state_round_trips_to_dict() -> None:
    state = AgentState(
        name="critic",
        step=2,
        actions=[{"arguments": {"ticket": "INC-1"}, "name": "create_ticket", "step": 1}],
        messages=[{"content": "Looks risky.", "role": "assistant"}],
        reasoning=[{"content": "Check assumptions.", "step": 1}],
        metadata={"model": "strong"},
    )

    restored = AgentState.from_dict(state.to_dict())

    assert restored == state
    assert restored.messages is not state.messages
    assert restored.metadata is not state.metadata


def test_agent_state_is_exported_from_package_root() -> None:
    assert an.AgentState is AgentState


def test_graph_state_tracks_agent_states_by_name() -> None:
    graph = GraphState(run_id="run-1")
    planner = AgentState(name="planner")

    graph.set_agent_state(planner)

    assert graph.agent_states == {"planner": planner}
    assert graph.get_agent_state("planner") is planner


def test_graph_state_round_trips_to_dict() -> None:
    graph = GraphState(
        run_id="run-2",
        agent_states={"critic": AgentState(name="critic", step=1)},
        metadata={"trace_id": "trace-1"},
    )

    restored = GraphState.from_dict(graph.to_dict())

    assert restored == graph
    assert restored.agent_states is not graph.agent_states
    assert restored.agent_states["critic"] is not graph.agent_states["critic"]
    assert restored.metadata is not graph.metadata


def test_graph_state_is_exported_from_package_root() -> None:
    assert an.GraphState is GraphState
