import pytest

import agentnet as an


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


class SourceModule(an.Module):
    interface = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.KeyValueRepresentation(required_keys=["summary"])],
    )

    async def arun(self, input: object, context: object | None = None) -> object:
        return {"summary": str(input)}


class TargetModule(an.Module):
    input_interface = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.KeyValueRepresentation(required_keys=["summary"])],
    )

    async def arun(self, input: object, context: object | None = None) -> str:
        self.input_interface.validate(input, representation="key_value")
        assert isinstance(input, dict)
        return str(input["summary"]).upper()


def test_graph_validator_accepts_valid_compiled_graph() -> None:
    first = NamedModule("first")
    second = NamedModule("second")
    compiled = an.compile_graph(an.Sequential(first, second))

    result = an.validate_graph(compiled)

    assert result is compiled


def test_graph_validator_rejects_unknown_edge_target() -> None:
    first = NamedModule("first")
    compiled = an.CompiledGraph(
        nodes={"first": first},
        edges={"first": ("missing",)},
        entry_nodes=("first",),
        output_nodes=("first",),
    )

    with pytest.raises(an.AgentNetValidationError, match="missing"):
        an.validate_graph(compiled)


def test_graph_validator_rejects_cycles() -> None:
    first = NamedModule("first")
    second = NamedModule("second")
    compiled = an.CompiledGraph(
        nodes={"first": first, "second": second},
        edges={"first": ("second",), "second": ("first",)},
        entry_nodes=("first",),
        output_nodes=("second",),
    )

    with pytest.raises(an.AgentNetValidationError, match="cycle"):
        an.validate_graph(compiled)


def test_graph_validator_rejects_unreachable_nodes() -> None:
    first = NamedModule("first")
    orphan = NamedModule("orphan")
    compiled = an.CompiledGraph(
        nodes={"first": first, "orphan": orphan},
        edges={"first": (), "orphan": ()},
        entry_nodes=("first",),
        output_nodes=("first",),
    )

    with pytest.raises(an.AgentNetValidationError, match="unreachable"):
        an.validate_graph(compiled)


def test_graph_validator_accepts_compatible_agent_interfaces() -> None:
    source = an.ReActAgent(
        "source",
        interface=an.Interface(
            semantic_contract=an.SemanticContract(required_fields=["summary"]),
            representations=[an.Representation("json")],
        ),
    )
    target = an.ReActAgent(
        "target",
        input_interface=an.Interface(
            semantic_contract=an.SemanticContract(required_fields=["summary"]),
            representations=[an.Representation("json")],
        ),
    )

    compiled = an.validate_graph(an.Sequential(source, target))

    assert compiled.edges == {"source": ("target",), "target": ()}


def test_graph_validator_rejects_incompatible_agent_interfaces() -> None:
    source = an.ReActAgent(
        "source",
        interface=an.Interface(
            semantic_contract=an.SemanticContract(required_fields=["summary"]),
            representations=[an.Representation("json")],
        ),
    )
    target = an.ReActAgent(
        "target",
        input_interface=an.Interface(
            semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
            representations=[an.Representation("json")],
        ),
    )

    with pytest.raises(an.AgentNetValidationError, match="risks"):
        an.validate_graph(an.Sequential(source, target))


def test_graph_communication_validates_and_executes_end_to_end() -> None:
    graph = an.Sequential(SourceModule("source"), TargetModule("target"))
    compiled = an.validate_graph(graph)

    assert compiled.edges == {"source": ("target",), "target": ()}
    assert an.run(graph, "ready") == "READY"


def test_graph_validator_is_exported_from_package_root() -> None:
    from agentnet.graphs import GraphValidator, validate_graph

    assert an.GraphValidator is GraphValidator
    assert an.validate_graph is validate_graph
