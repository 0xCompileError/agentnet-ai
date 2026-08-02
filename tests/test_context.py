import agentnet as an
from agentnet.core.context import RunContext
from agentnet.core.state import AgentState, GraphState


def test_run_context_creates_graph_state_for_run() -> None:
    context = RunContext(run_id="run-1")

    assert context.run_id == "run-1"
    assert context.graph_state == GraphState(run_id="run-1")


def test_run_context_round_trips_to_dict() -> None:
    graph_state = GraphState(
        run_id="run-2",
        agent_states={"planner": AgentState(name="planner", step=1)},
    )
    context = RunContext(
        run_id="run-2",
        graph_state=graph_state,
        metadata={"request_id": "req-1"},
    )

    restored = RunContext.from_dict(context.to_dict())

    assert restored == context
    assert restored.graph_state is not context.graph_state
    assert restored.metadata is not context.metadata


def test_run_context_can_be_cancelled() -> None:
    context = RunContext(run_id="run-3")

    context.cancel()

    assert context.cancelled is True
    assert RunContext.from_dict(context.to_dict()).cancelled is True


def test_run_context_is_exported_from_package_root() -> None:
    assert an.RunContext is RunContext
