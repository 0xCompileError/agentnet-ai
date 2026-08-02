import agentnet as an


def test_runtime_public_contract_exports_core_primitives() -> None:
    assert an.Module("module").state_dict() == {"name": "module"}
    assert an.AgentState(name="agent").name == "agent"
    assert an.GraphState(run_id="run").run_id == "run"
    assert an.RunContext(run_id="run").graph_state.run_id == "run"
    assert an.GraphResult(output=None, graph_state=an.GraphState(run_id="run")).succeeded is True
