import pytest

import agentnet as an


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return input


def test_topology_search_space_generates_bounded_mutation_candidates() -> None:
    base = an.Sequential(NamedModule("a"), NamedModule("b"), name="flow")
    search_space = an.TopologySearchSpace(
        branch_candidates=[NamedModule("critic")],
        replacement_candidates=[NamedModule("writer")],
        allowed_mutations=["branch_insertion", "node_replacement"],
        max_trials=2,
    )

    candidates = list(an.TopologyMutationEngine().generate(base, search_space))
    assert all(candidate.mutation is not None for candidate in candidates)

    assert [candidate.mutation.kind for candidate in candidates if candidate.mutation] == [
        "branch_insertion",
        "node_replacement",
    ]
    assert search_space.to_dict()["max_trials"] == 2


def test_branch_insertion_and_removal_mutations() -> None:
    engine = an.TopologyMutationEngine()
    parallel = an.Parallel(NamedModule("a"), NamedModule("b"), name="experts")

    inserted = engine.insert_branch(parallel, NamedModule("c"))

    assert isinstance(inserted.module, an.Parallel)
    assert [module.name for module in inserted.module.modules] == ["a", "b", "c"]
    assert inserted.mutation is not None
    assert inserted.mutation.kind == "branch_insertion"
    assert inserted.mutation.inserted == "c"

    removed = engine.remove_branch(inserted.module, "b")

    assert isinstance(removed.module, an.Parallel)
    assert [module.name for module in removed.module.modules] == ["a", "c"]
    assert removed.mutation is not None
    assert removed.mutation.kind == "branch_removal"
    assert removed.mutation.removed == "b"


def test_router_and_reducer_insertion_mutations() -> None:
    engine = an.TopologyMutationEngine()
    base = NamedModule("worker")
    router_module = NamedModule("route")

    routed = engine.insert_router(
        base,
        router=router_module,
        routes={"default": base},
        fallback=NamedModule("fallback"),
        name="gate",
    )

    assert isinstance(routed.module, an.Router)
    assert routed.module.router is router_module
    assert routed.module.routes == {"default": base}
    assert routed.mutation is not None
    assert routed.mutation.kind == "router_insertion"

    reduced = engine.insert_reducer(
        an.Parallel(NamedModule("left"), NamedModule("right"), name="branches"),
        reducer=NamedModule("merge"),
    )

    assert isinstance(reduced.module, an.Parallel)
    assert reduced.module.reducer is not None
    assert reduced.module.reducer.name == "merge"
    assert reduced.mutation is not None
    assert reduced.mutation.kind == "reducer_insertion"


def test_node_replacement_mutation_preserves_other_nodes() -> None:
    base = an.Sequential(NamedModule("planner"), NamedModule("writer"), name="flow")

    candidate = an.TopologyMutationEngine().replace_node(
        base,
        target_name="writer",
        replacement=NamedModule("critic"),
    )

    assert isinstance(candidate.module, an.Sequential)
    assert [module.name for module in candidate.module.modules] == ["planner", "critic"]
    assert candidate.mutation is not None
    assert candidate.mutation.kind == "node_replacement"
    assert candidate.mutation.removed == "writer"
    assert candidate.mutation.inserted == "critic"


def test_architecture_scorer_reports_complexity_metrics() -> None:
    graph = an.compile_graph(an.Sequential(NamedModule("a"), NamedModule("b")))
    scorer = an.ArchitectureScorer(
        base_scorer=lambda candidate: 10.0,
        node_penalty=0.5,
        branch_penalty=0.25,
        depth_penalty=0.1,
    )

    score = scorer.score(graph)

    assert score.score == pytest.approx(8.55)
    assert score.metrics == {
        "architecture.base_score": 10.0,
        "architecture.branch_count": 1.0,
        "architecture.depth": 2.0,
        "architecture.node_count": 2.0,
        "architecture.penalty": 1.45,
    }
    assert an.ArchitectureScore.from_dict(score.to_dict()).to_dict() == score.to_dict()


def test_topology_optimizer_search_records_trials_and_checkpoints() -> None:
    base = NamedModule("base")
    optimizer = an.TopologyOptimizer(
        search_space=an.TopologySearchSpace(
            branch_candidates=[NamedModule("critic")],
            allowed_mutations=["branch_insertion"],
            max_trials=2,
        ),
        metadata={"optimizer": "topology-search"},
    )

    result = optimizer.search(
        base,
        scorer=an.ArchitectureScorer(base_scorer=lambda graph: float(len(graph.nodes))),
    )

    assert isinstance(result.module, an.Parallel)
    assert result.mutation is not None
    assert result.mutation.kind == "branch_insertion"
    assert result.score == 2.0
    assert result.metadata["optimizer"] == "topology-search"
    assert result.metadata["evaluated_candidates"] == 2
    assert [checkpoint.trial for checkpoint in result.checkpoints] == [1, 2]
    assert result.checkpoints[1].mutation is not None
    assert result.checkpoints[1].mutation.kind == "branch_insertion"
    serialized = result.checkpoints[1].to_dict()
    assert "module" not in serialized
    assert serialized["graph_descriptor"]["nodes"] == [
        {"name": "base", "type": "NamedModule"},
        {"name": "critic", "type": "NamedModule"},
    ]


def test_topology_checkpoint_round_trips_descriptor_only() -> None:
    graph = an.compile_graph(an.Sequential(NamedModule("a"), NamedModule("b")))
    score = an.ArchitectureScore(score=0.75, metrics={"quality": 0.75})
    mutation = an.TopologyMutation(
        kind="node_replacement",
        target="b",
        inserted="critic",
        removed="b",
    )
    checkpoint = an.TopologyCheckpoint(
        trial=3,
        score=score,
        compiled_graph=graph,
        mutation=mutation,
        metadata={"phase": "local-search"},
    )

    serialized = checkpoint.to_dict()

    assert "compiled_graph" not in serialized
    assert serialized["mutation"] == mutation.to_dict()
    assert an.TopologyCheckpoint.from_dict(serialized).to_dict() == serialized


def test_topology_search_public_exports_are_available() -> None:
    assert an.ArchitectureScore is not None
    assert an.ArchitectureScorer is not None
    assert an.TopologyCandidate is not None
    assert an.TopologyCheckpoint is not None
    assert an.TopologyMutation is not None
    assert an.TopologyMutationEngine is not None
