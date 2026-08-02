import json

import agentnet as an
from agentnet.constraints import Constraint, ConstraintDescriptor


class NonEmptyConstraint(Constraint):
    def check(self, candidate: object, context: object | None = None) -> bool:
        return bool(candidate)


def test_constraint_serializes_to_descriptor_dict() -> None:
    constraint = NonEmptyConstraint(
        "non_empty",
        description="Value must be present.",
        kind=an.ConstraintKind.SOFT,
        metadata={"category": "presence"},
    )

    serialized = constraint.to_dict()

    assert serialized == {
        "children": [],
        "description": "Value must be present.",
        "kind": "soft",
        "metadata": {"category": "presence"},
        "name": "non_empty",
        "parameters": {},
        "type": "NonEmptyConstraint",
        "version": "1",
    }
    assert ConstraintDescriptor.from_dict(serialized).to_dict() == serialized


def test_constraint_serialization_preserves_explicit_version() -> None:
    constraint = NonEmptyConstraint("non_empty", version="2026-07")

    serialized = constraint.to_dict()

    assert constraint.version == "2026-07"
    assert serialized["version"] == "2026-07"
    assert ConstraintDescriptor.from_dict(serialized).version == "2026-07"


def test_constraint_descriptor_defaults_missing_version() -> None:
    descriptor = ConstraintDescriptor.from_dict(
        {
            "children": [],
            "description": None,
            "kind": "hard",
            "metadata": {},
            "name": "legacy",
            "parameters": {},
            "type": "LegacyConstraint",
        }
    )

    assert descriptor.version == "1"


def test_composite_constraint_serializes_children() -> None:
    constraint = NonEmptyConstraint("left") & NonEmptyConstraint("right")

    serialized = constraint.to_dict()

    assert serialized["type"] == "AndConstraint"
    assert serialized["parameters"] == {"operator": "and"}
    assert [child["name"] for child in serialized["children"]] == ["left", "right"]


def test_scoped_constraints_serialize_scope_parameters_and_inner_constraint() -> None:
    constraint = an.EdgeConstraint(
        "first",
        "second",
        NonEmptyConstraint("edge_payload"),
    )

    serialized = constraint.to_dict()

    assert serialized["parameters"] == {"source": "first", "target": "second"}
    assert serialized["children"][0]["name"] == "edge_payload"


def test_builtin_constraints_serialize_json_data_descriptors() -> None:
    def always_pass(candidate: object, context: object | None = None) -> bool:
        return True

    constraints: list[tuple[Constraint, dict[str, object]]] = [
        (
            an.CostConstraint(0.25),
            {"field": "cost", "max_cost": 0.25},
        ),
        (
            an.TokenConstraint(100),
            {"field": "tokens", "max_tokens": 100},
        ),
        (
            an.LatencyConstraint(2500),
            {"field": "latency_ms", "max_latency_ms": 2500.0},
        ),
        (
            an.MemoryConstraint(128),
            {"field": "memory_mb", "max_memory_mb": 128.0},
        ),
        (
            an.SafetyConstraint(["secret"]),
            {
                "blocked_terms": ["secret"],
                "case_sensitive": False,
                "field": "content",
            },
        ),
        (
            an.TopologyConstraint(
                allowed_modules=["Sequential"],
                max_branches=1,
                max_depth=2,
                max_nodes=3,
            ),
            {
                "allowed_modules": ["Sequential"],
                "max_branches": 1,
                "max_depth": 2,
                "max_nodes": 3,
            },
        ),
        (
            an.RetryConstraint(
                allowed_backoff=["exponential"],
                max_quality_retries=1,
                max_total_attempts=3,
                max_transport_retries=2,
            ),
            {
                "allowed_backoff": ["exponential"],
                "max_quality_retries": 1,
                "max_total_attempts": 3,
                "max_transport_retries": 2,
            },
        ),
        (
            an.ToolConstraint(["search_docs"]),
            {"allowed": ["search_docs"], "field": "tools"},
        ),
        (
            an.ModelConstraint(["fast"]),
            {"allowed": ["fast"], "field": "model"},
        ),
        (
            an.RepresentationConstraint(["json"]),
            {"allowed": ["json"], "field": "representation"},
        ),
        (
            an.SchemaConstraint(list[str], label="payload"),
            {"label": "payload", "schema": "list[str]"},
        ),
        (
            an.CustomConstraint("custom", always_pass),
            {"custom": True},
        ),
    ]

    for constraint, expected_parameters in constraints:
        serialized = constraint.to_dict()

        json.dumps(serialized, sort_keys=True)
        assert serialized["parameters"] == expected_parameters
        assert ConstraintDescriptor.from_dict(serialized).to_dict() == serialized

    assert "always_pass" not in str(constraints[-1][0].to_dict())


def test_nested_constraint_descriptor_round_trips_without_rehydration() -> None:
    constraint = an.GraphConstraint(an.TopologyConstraint(max_nodes=2)) & an.NodeConstraint(
        "worker",
        NonEmptyConstraint("node_payload"),
    )

    serialized = constraint.to_dict()
    descriptor = ConstraintDescriptor.from_dict(serialized)

    assert descriptor.to_dict() == serialized
    assert descriptor.type == "AndConstraint"
    assert [child.type for child in descriptor.children] == [
        "GraphConstraint",
        "NodeConstraint",
    ]


def test_constraint_descriptor_is_exported_from_package_root() -> None:
    assert an.ConstraintDescriptor is ConstraintDescriptor
