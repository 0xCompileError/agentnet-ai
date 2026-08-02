"""Constraint-aware topology search primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.constraints import (
    Constraint,
    ConstraintResult,
    validate_training_constraints,
)
from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetValidationError,
    Module,
)
from agentnet.graphs import (
    CompiledGraph,
    Parallel,
    Reducer,
    Router,
    Sequential,
    validate_graph,
)
from agentnet.mcp._security import (
    validate_descriptor_payload_no_secrets,
    validate_safe_metadata,
)

_MUTATION_KINDS = (
    "branch_insertion",
    "branch_removal",
    "router_insertion",
    "reducer_insertion",
    "node_replacement",
)


@dataclass(frozen=True, slots=True, init=False)
class TopologySearchSpace:
    """Bounds and explicit building blocks for topology candidate search."""

    allowed_modules: tuple[str, ...]
    max_nodes: int | None
    max_branches: int | None
    max_depth: int | None
    max_trials: int | None
    allowed_mutations: tuple[str, ...]
    branch_candidates: tuple[Module, ...]
    replacement_candidates: tuple[Module, ...]
    router_candidates: tuple[Module, ...]
    reducer_candidates: tuple[Module, ...]

    def __init__(
        self,
        *,
        allowed_modules: Sequence[str] | None = None,
        max_nodes: int | None = None,
        max_branches: int | None = None,
        max_depth: int | None = None,
        max_trials: int | None = None,
        allowed_mutations: Sequence[str] | None = None,
        branch_candidates: Sequence[Module] | None = None,
        replacement_candidates: Sequence[Module] | None = None,
        router_candidates: Sequence[Module] | None = None,
        reducer_candidates: Sequence[Module] | None = None,
    ) -> None:
        _validate_positive("max_nodes", max_nodes)
        _validate_positive("max_branches", max_branches)
        _validate_positive("max_depth", max_depth)
        _validate_positive("max_trials", max_trials)

        mutation_kinds = tuple(allowed_mutations or _MUTATION_KINDS)
        unknown_mutations = tuple(
            mutation for mutation in mutation_kinds if mutation not in _MUTATION_KINDS
        )
        if unknown_mutations:
            raise AgentNetConfigurationError(
                f"Unknown topology mutation kind {unknown_mutations[0]!r}"
            )

        object.__setattr__(self, "allowed_modules", tuple(allowed_modules or ()))
        object.__setattr__(self, "max_nodes", max_nodes)
        object.__setattr__(self, "max_branches", max_branches)
        object.__setattr__(self, "max_depth", max_depth)
        object.__setattr__(self, "max_trials", max_trials)
        object.__setattr__(self, "allowed_mutations", mutation_kinds)
        object.__setattr__(
            self,
            "branch_candidates",
            _module_tuple(branch_candidates or (), "branch_candidates"),
        )
        object.__setattr__(
            self,
            "replacement_candidates",
            _module_tuple(replacement_candidates or (), "replacement_candidates"),
        )
        object.__setattr__(
            self,
            "router_candidates",
            _module_tuple(router_candidates or (), "router_candidates"),
        )
        object.__setattr__(
            self,
            "reducer_candidates",
            _module_tuple(reducer_candidates or (), "reducer_candidates"),
        )

    def violation(self, graph: CompiledGraph) -> str | None:
        if self.allowed_modules:
            allowed = set(self.allowed_modules)
            for module in graph.nodes.values():
                module_type = module.__class__.__name__
                if module_type not in allowed:
                    return f"Module type {module_type!r} is outside the search space"

        if self.max_nodes is not None and len(graph.nodes) > self.max_nodes:
            return f"Graph has {len(graph.nodes)} nodes, exceeding {self.max_nodes}"

        if self.max_branches is not None:
            max_branches = _graph_branch_count(graph)
            if max_branches > self.max_branches:
                return (
                    f"Graph has branch count {max_branches}, "
                    f"exceeding {self.max_branches}"
                )

        if self.max_depth is not None:
            depth = _graph_depth(graph)
            if depth > self.max_depth:
                return f"Graph depth is {depth}, exceeding {self.max_depth}"

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_modules": list(self.allowed_modules),
            "allowed_mutations": list(self.allowed_mutations),
            "branch_candidates": [
                _module_descriptor(module) for module in self.branch_candidates
            ],
            "max_branches": self.max_branches,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_trials": self.max_trials,
            "reducer_candidates": [
                _module_descriptor(module) for module in self.reducer_candidates
            ],
            "replacement_candidates": [
                _module_descriptor(module) for module in self.replacement_candidates
            ],
            "router_candidates": [
                _module_descriptor(module) for module in self.router_candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class TopologyMutation:
    """Descriptor for one topology mutation."""

    kind: str
    target: str | None = None
    inserted: str | None = None
    removed: str | None = None
    rationale: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _MUTATION_KINDS:
            raise AgentNetConfigurationError(
                f"TopologyMutation kind must be one of: {', '.join(_MUTATION_KINDS)}"
            )
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="TopologyMutation")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "rationale": self.rationale,
            "removed": self.removed,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, mutation: Mapping[str, Any]) -> Self:
        return cls(
            kind=str(mutation["kind"]),
            target=None if mutation.get("target") is None else str(mutation["target"]),
            inserted=(
                None if mutation.get("inserted") is None else str(mutation["inserted"])
            ),
            removed=None if mutation.get("removed") is None else str(mutation["removed"]),
            rationale=(
                None if mutation.get("rationale") is None else str(mutation["rationale"])
            ),
            metadata=dict(mutation.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TopologyCandidate:
    """Candidate module paired with the mutation that produced it."""

    module: Module
    mutation: TopologyMutation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module, Module):
            raise AgentNetConfigurationError("TopologyCandidate module must be a Module")


class TopologyMutationEngine:
    """Generate local graph mutations from explicit module candidates."""

    def generate(
        self,
        module: Module,
        search_space: TopologySearchSpace,
    ) -> Iterable[TopologyCandidate]:
        yielded = 0

        def emit(candidate: TopologyCandidate) -> TopologyCandidate | None:
            nonlocal yielded
            if search_space.max_trials is not None and yielded >= search_space.max_trials:
                return None
            yielded += 1
            return candidate

        for kind in search_space.allowed_mutations:
            if kind == "branch_insertion":
                for branch in search_space.branch_candidates:
                    candidate = emit(self.insert_branch(module, branch))
                    if candidate is None:
                        return
                    yield candidate
            elif kind == "branch_removal":
                if isinstance(module, Parallel):
                    for branch in module.modules:
                        candidate = emit(self.remove_branch(module, branch.name))
                        if candidate is None:
                            return
                        yield candidate
            elif kind == "router_insertion":
                for router in search_space.router_candidates:
                    candidate = emit(
                        self.insert_router(
                            module,
                            router=router,
                            routes={"default": module},
                        )
                    )
                    if candidate is None:
                        return
                    yield candidate
            elif kind == "reducer_insertion":
                for reducer in search_space.reducer_candidates:
                    candidate = emit(self.insert_reducer(module, reducer=reducer))
                    if candidate is None:
                        return
                    yield candidate
            elif kind == "node_replacement":
                graph = validate_graph(module)
                for target_name in graph.nodes:
                    for replacement in search_space.replacement_candidates:
                        candidate = emit(
                            self.replace_node(
                                module,
                                target_name=target_name,
                                replacement=replacement,
                            )
                        )
                        if candidate is None:
                            return
                        yield candidate

    def insert_branch(
        self,
        module: Module,
        branch: Module,
        *,
        name: str | None = None,
    ) -> TopologyCandidate:
        if not isinstance(branch, Module):
            raise AgentNetConfigurationError("Branch insertion requires a Module branch")
        if isinstance(module, Parallel):
            candidate = Parallel(
                *module.modules,
                branch,
                reducer=module.reducer,
                name=name or module.name,
            )
            target = module.name
        else:
            candidate = Parallel(
                module,
                branch,
                name=name or f"{module.name}.parallel",
            )
            target = module.name
        return TopologyCandidate(
            candidate,
            TopologyMutation(
                kind="branch_insertion",
                target=target,
                inserted=branch.name,
                rationale="Inserted parallel branch candidate.",
            ),
        )

    def remove_branch(self, module: Parallel, branch_name: str) -> TopologyCandidate:
        if not isinstance(module, Parallel):
            raise AgentNetConfigurationError("Branch removal requires a Parallel module")
        remaining = tuple(branch for branch in module.modules if branch.name != branch_name)
        if len(remaining) == len(module.modules):
            raise AgentNetValidationError(f"Branch {branch_name!r} was not found")
        if not remaining:
            raise AgentNetValidationError("Cannot remove the final parallel branch")
        return TopologyCandidate(
            Parallel(*remaining, reducer=module.reducer, name=module.name),
            TopologyMutation(
                kind="branch_removal",
                target=module.name,
                removed=branch_name,
                rationale="Removed parallel branch candidate.",
            ),
        )

    def insert_router(
        self,
        module: Module,
        *,
        router: Module,
        routes: Mapping[str, Module] | None = None,
        fallback: Module | None = None,
        name: str = "router",
    ) -> TopologyCandidate:
        route_map = dict(routes or {"default": module})
        candidate = Router(
            router=router,
            routes=route_map,
            fallback=fallback,
            name=name,
        )
        return TopologyCandidate(
            candidate,
            TopologyMutation(
                kind="router_insertion",
                target=module.name,
                inserted=name,
                rationale="Inserted router candidate.",
            ),
        )

    def insert_reducer(
        self,
        module: Module,
        *,
        reducer: Module,
        name: str | None = None,
    ) -> TopologyCandidate:
        if isinstance(module, Parallel):
            candidate = Parallel(*module.modules, reducer=reducer, name=name or module.name)
            target = module.name
        else:
            candidate = Parallel(module, reducer=reducer, name=name or f"{module.name}.reduced")
            target = module.name
        return TopologyCandidate(
            candidate,
            TopologyMutation(
                kind="reducer_insertion",
                target=target,
                inserted=reducer.name,
                rationale="Inserted reducer candidate.",
            ),
        )

    def replace_node(
        self,
        module: Module,
        *,
        target_name: str,
        replacement: Module,
    ) -> TopologyCandidate:
        candidate, replaced = _replace_node(module, target_name, replacement)
        if not replaced:
            raise AgentNetValidationError(f"Node {target_name!r} was not found")
        return TopologyCandidate(
            candidate,
            TopologyMutation(
                kind="node_replacement",
                target=target_name,
                inserted=replacement.name,
                removed=target_name,
                rationale="Replaced topology node candidate.",
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchitectureScore:
    """Score and metrics for one compiled architecture."""

    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="ArchitectureScore")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, score: Mapping[str, Any]) -> Self:
        return cls(
            score=float(score["score"]),
            metrics={
                str(key): float(value)
                for key, value in dict(score.get("metrics", {})).items()
            },
            metadata=dict(score.get("metadata", {})),
        )


class ArchitectureScorer:
    """Deterministic architecture scoring with optional complexity penalties."""

    def __init__(
        self,
        *,
        base_scorer: Callable[[CompiledGraph], float] | None = None,
        node_penalty: float = 0.0,
        branch_penalty: float = 0.0,
        depth_penalty: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if base_scorer is not None and not callable(base_scorer):
            raise AgentNetConfigurationError("ArchitectureScorer base_scorer must be callable")
        for name, value in (
            ("node_penalty", node_penalty),
            ("branch_penalty", branch_penalty),
            ("depth_penalty", depth_penalty),
        ):
            if value < 0:
                raise AgentNetConfigurationError(f"{name} cannot be negative")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="ArchitectureScorer")
        self.base_scorer = base_scorer
        self.node_penalty = float(node_penalty)
        self.branch_penalty = float(branch_penalty)
        self.depth_penalty = float(depth_penalty)
        self.metadata = metadata_copy

    def __call__(self, graph: CompiledGraph) -> ArchitectureScore:
        return self.score(graph)

    def score(self, graph: CompiledGraph) -> ArchitectureScore:
        base_score = 0.0 if self.base_scorer is None else float(self.base_scorer(graph))
        node_count = len(graph.nodes)
        branch_count = _graph_branch_count(graph)
        depth = _graph_depth(graph)
        penalty = (
            node_count * self.node_penalty
            + branch_count * self.branch_penalty
            + depth * self.depth_penalty
        )
        return ArchitectureScore(
            score=base_score - penalty,
            metrics={
                "architecture.base_score": base_score,
                "architecture.branch_count": float(branch_count),
                "architecture.depth": float(depth),
                "architecture.node_count": float(node_count),
                "architecture.penalty": penalty,
            },
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True, init=False)
class TopologyCheckpoint:
    """Descriptor-only record for one topology search trial."""

    trial: int
    score: ArchitectureScore
    graph_descriptor: Mapping[str, Any]
    mutation: TopologyMutation | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        trial: int,
        score: ArchitectureScore,
        compiled_graph: CompiledGraph | None = None,
        graph_descriptor: Mapping[str, Any] | None = None,
        mutation: TopologyMutation | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if trial < 1:
            raise AgentNetConfigurationError("TopologyCheckpoint trial must be at least 1")
        if compiled_graph is None and graph_descriptor is None:
            raise AgentNetConfigurationError(
                "TopologyCheckpoint requires compiled_graph or graph_descriptor"
            )
        descriptor = (
            _graph_descriptor(compiled_graph)
            if compiled_graph is not None
            else dict(graph_descriptor or {})
        )
        validate_descriptor_payload_no_secrets(descriptor, label="TopologyCheckpoint")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TopologyCheckpoint")
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "graph_descriptor", descriptor)
        object.__setattr__(self, "mutation", mutation)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_descriptor": dict(self.graph_descriptor),
            "metadata": dict(self.metadata),
            "mutation": None if self.mutation is None else self.mutation.to_dict(),
            "score": self.score.to_dict(),
            "trial": self.trial,
        }

    @classmethod
    def from_dict(cls, checkpoint: Mapping[str, Any]) -> Self:
        mutation = checkpoint.get("mutation")
        return cls(
            trial=int(checkpoint["trial"]),
            score=ArchitectureScore.from_dict(dict(checkpoint["score"])),
            graph_descriptor=dict(checkpoint["graph_descriptor"]),
            mutation=(
                None
                if mutation is None
                else TopologyMutation.from_dict(dict(mutation))
            ),
            metadata=dict(checkpoint.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TopologyOptimizationResult:
    """Best topology selected by topology search."""

    module: Module
    compiled_graph: CompiledGraph
    score: float
    constraint_results: tuple[ConstraintResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    mutation: TopologyMutation | None = None
    architecture_score: ArchitectureScore | None = None
    checkpoints: tuple[TopologyCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="TopologyOptimizationResult")
        object.__setattr__(
            self,
            "constraint_results",
            tuple(self.constraint_results),
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))


TopologyScorer = Callable[[CompiledGraph], float | ArchitectureScore]


class TopologyOptimizer:
    """Search topology candidates within explicit bounds and constraints."""

    def __init__(
        self,
        *,
        search_space: TopologySearchSpace | None = None,
        constraints: Iterable[Constraint] | None = None,
        metadata: Mapping[str, Any] | None = None,
        mutation_engine: TopologyMutationEngine | None = None,
    ) -> None:
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TopologyOptimizer")
        self.search_space = search_space or TopologySearchSpace()
        self.constraints = tuple(constraints or ())
        self.metadata = metadata_copy
        self.mutation_engine = mutation_engine or TopologyMutationEngine()

    def search(
        self,
        seed: Module,
        *,
        scorer: TopologyScorer,
    ) -> TopologyOptimizationResult:
        """Generate local topology mutations from a seed and select the best candidate."""

        candidates = [TopologyCandidate(seed)]
        candidates.extend(self.mutation_engine.generate(seed, self.search_space))
        return self.optimize(candidates, scorer=scorer)

    def optimize(
        self,
        candidates: Iterable[Module | TopologyCandidate],
        *,
        scorer: TopologyScorer,
    ) -> TopologyOptimizationResult:
        best: TopologyOptimizationResult | None = None
        checkpoints: list[TopologyCheckpoint] = []
        evaluated_candidates = 0
        rejected_candidates = 0

        for raw_candidate in candidates:
            if (
                self.search_space.max_trials is not None
                and evaluated_candidates + rejected_candidates >= self.search_space.max_trials
            ):
                break

            candidate = _coerce_candidate(raw_candidate)
            trial = evaluated_candidates + rejected_candidates + 1
            try:
                graph = validate_graph(candidate.module)
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            if self.search_space.violation(graph) is not None:
                rejected_candidates += 1
                continue

            candidate_metadata: dict[str, Any] = {}
            try:
                constraint_results = validate_training_constraints(
                    graph,
                    self.constraints,
                    metadata=candidate_metadata,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            evaluated_candidates += 1
            architecture_score = _coerce_architecture_score(scorer(graph))
            checkpoint = TopologyCheckpoint(
                trial=trial,
                score=architecture_score,
                compiled_graph=graph,
                mutation=candidate.mutation,
                metadata={"accepted": True},
            )
            checkpoints.append(checkpoint)
            result = TopologyOptimizationResult(
                module=candidate.module,
                compiled_graph=graph,
                score=architecture_score.score,
                constraint_results=constraint_results,
                metadata={
                    **self.metadata,
                    "evaluated_candidates": evaluated_candidates,
                    "rejected_candidates": rejected_candidates,
                    "training_constraint_results": candidate_metadata.get(
                        "training_constraint_results",
                        [],
                    ),
                },
                mutation=candidate.mutation,
                architecture_score=architecture_score,
                checkpoints=tuple(checkpoints),
            )
            if best is None or result.score > best.score:
                best = result

        if best is None:
            raise AgentNetValidationError(
                "No topology candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        object.__setattr__(best, "checkpoints", tuple(checkpoints))
        return best


def _validate_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise AgentNetConfigurationError(f"{name} must be positive")


def _module_tuple(modules: Iterable[Module], field_name: str) -> tuple[Module, ...]:
    module_tuple = tuple(modules)
    if any(not isinstance(module, Module) for module in module_tuple):
        raise AgentNetConfigurationError(f"{field_name} entries must be Module instances")
    return module_tuple


def _coerce_candidate(candidate: Module | TopologyCandidate) -> TopologyCandidate:
    if isinstance(candidate, TopologyCandidate):
        return candidate
    if isinstance(candidate, Module):
        return TopologyCandidate(candidate)
    raise AgentNetConfigurationError("Topology candidates must be Module instances")


def _coerce_architecture_score(score: float | ArchitectureScore) -> ArchitectureScore:
    if isinstance(score, ArchitectureScore):
        return score
    return ArchitectureScore(
        score=float(score),
        metrics={"architecture.base_score": float(score)},
    )


def _replace_node(
    module: Module,
    target_name: str,
    replacement: Module,
) -> tuple[Module, bool]:
    if module.name == target_name:
        return replacement, True

    if isinstance(module, Sequential):
        changed = False
        modules: list[Module] = []
        for child in module.modules:
            new_child, replaced = _replace_node(child, target_name, replacement)
            modules.append(new_child)
            changed = changed or replaced
        if changed:
            return Sequential(*modules, name=module.name), True
        return module, False

    if isinstance(module, Parallel):
        changed = False
        modules = []
        for child in module.modules:
            new_child, replaced = _replace_node(child, target_name, replacement)
            modules.append(new_child)
            changed = changed or replaced
        reducer = module.reducer
        if reducer is not None:
            new_reducer, replaced = _replace_node(reducer, target_name, replacement)
            reducer = new_reducer
            changed = changed or replaced
        if changed:
            return Parallel(*modules, reducer=reducer, name=module.name), True
        return module, False

    if isinstance(module, Router):
        router, router_replaced = _replace_node(module.router, target_name, replacement)
        changed = router_replaced
        routes: dict[str, Module] = {}
        for route_name, route in module.routes.items():
            new_route, route_replaced = _replace_node(route, target_name, replacement)
            routes[route_name] = new_route
            changed = changed or route_replaced
        fallback = module.fallback
        if fallback is not None:
            new_fallback, fallback_replaced = _replace_node(
                fallback,
                target_name,
                replacement,
            )
            fallback = new_fallback
            changed = changed or fallback_replaced
        if changed:
            return Router(
                router=router,
                routes=routes,
                fallback=fallback,
                name=module.name,
            ), True
        return module, False

    if isinstance(module, Reducer):
        reducer, replaced = _replace_node(module.reducer, target_name, replacement)
        if replaced:
            return Reducer(reducer, name=module.name), True
        return module, False

    return module, False


def _graph_depth(graph: CompiledGraph) -> int:
    depths: dict[str, int] = {}

    def visit(node: str) -> int:
        if node in depths:
            return depths[node]
        targets = graph.edges.get(node, ())
        depth = 1 if not targets else 1 + max(visit(target) for target in targets)
        depths[node] = depth
        return depth

    return max((visit(node) for node in graph.entry_nodes), default=0)


def _graph_branch_count(graph: CompiledGraph) -> int:
    return max((len(targets) for targets in graph.edges.values()), default=0)


def _module_descriptor(module: Module) -> dict[str, str]:
    return {
        "name": module.name,
        "type": module.__class__.__name__,
    }


def _graph_descriptor(graph: CompiledGraph) -> dict[str, Any]:
    return {
        "edges": {source: list(targets) for source, targets in graph.edges.items()},
        "entry_nodes": list(graph.entry_nodes),
        "nodes": [_module_descriptor(module) for module in graph.nodes.values()],
        "output_nodes": list(graph.output_nodes),
    }
