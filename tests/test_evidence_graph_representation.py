import pytest

import agentnet as an


def test_evidence_graph_representation_accepts_nodes_and_edges() -> None:
    representation = an.EvidenceGraphRepresentation()

    value = {
        "nodes": [
            {"id": "claim", "text": "The release is ready."},
            {"id": "evidence", "text": "All validation passed."},
        ],
        "edges": [
            {"source": "evidence", "target": "claim", "relation": "supports"},
        ],
    }

    assert representation.identifier == "evidence_graph"
    assert representation.media_type == "application/vnd.agentnet.evidence-graph"
    assert representation.validate(value) == value


def test_evidence_graph_representation_requires_nodes() -> None:
    representation = an.EvidenceGraphRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload.nodes"):
        representation.validate({"edges": []}, label="payload")

    with pytest.raises(an.AgentNetValidationError, match="payload.nodes"):
        representation.validate({"nodes": []}, label="payload")


def test_evidence_graph_representation_validates_node_ids() -> None:
    representation = an.EvidenceGraphRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload.nodes\\[0\\].id"):
        representation.validate({"nodes": [{"text": "missing id"}]}, label="payload")

    with pytest.raises(an.AgentNetValidationError, match="duplicate"):
        representation.validate(
            {"nodes": [{"id": "claim"}, {"id": "claim"}]},
            label="payload",
        )


def test_evidence_graph_representation_validates_edge_references() -> None:
    representation = an.EvidenceGraphRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload.edges"):
        representation.validate(
            {"nodes": [{"id": "claim"}], "edges": "claim -> source"},
            label="payload",
        )

    with pytest.raises(an.AgentNetValidationError, match="unknown node"):
        representation.validate(
            {
                "nodes": [{"id": "claim"}],
                "edges": [{"source": "source", "target": "claim"}],
            },
            label="payload",
        )


def test_evidence_graph_representation_can_require_edges() -> None:
    representation = an.EvidenceGraphRepresentation(require_edges=True)

    with pytest.raises(an.AgentNetValidationError, match="at least one edge"):
        representation.validate({"nodes": [{"id": "claim"}]}, label="payload")


def test_evidence_graph_representation_rejects_non_mapping_payloads() -> None:
    representation = an.EvidenceGraphRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate(["claim"], label="payload")


def test_evidence_graph_representation_is_exported_from_package_root() -> None:
    assert an.EvidenceGraphRepresentation is not None
