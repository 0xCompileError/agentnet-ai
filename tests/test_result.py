import agentnet as an
from agentnet.core.result import GraphResult
from agentnet.core.state import AgentState, GraphState


def test_graph_result_carries_output_and_graph_state() -> None:
    graph_state = GraphState(
        run_id="run-1",
        agent_states={"writer": AgentState(name="writer", step=1)},
    )
    result = GraphResult(
        output={"answer": "done"},
        graph_state=graph_state,
        metadata={"latency_ms": 12},
    )

    assert result.output == {"answer": "done"}
    assert result.graph_state is graph_state
    assert result.succeeded is True
    assert result.metadata == {"latency_ms": 12}


def test_graph_result_round_trips_to_dict() -> None:
    result = GraphResult(
        output=["a", "b"],
        graph_state=GraphState(run_id="run-2"),
        succeeded=False,
        metadata={"error": "timeout"},
    )

    restored = GraphResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.graph_state is not result.graph_state
    assert isinstance(restored.output, list)


def test_graph_result_is_exported_from_package_root() -> None:
    assert an.GraphResult is GraphResult
