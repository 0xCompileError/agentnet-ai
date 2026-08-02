"""Bounded automatic candidate generation for the simple training API."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentnet.agents import ReActAgent
from agentnet.core import AgentNetConfigurationError, Module
from agentnet.graphs import DAG, Parallel, Reducer, Router, Sequential
from agentnet.llms import ChatRequest, FakeLLM
from agentnet.policies import RetryPolicy


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    """One executable candidate proposed by a training optimizer."""

    net: Module
    stage: str
    changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.net, Module):
            raise AgentNetConfigurationError("TrainingCandidate net must be a Module")
        if not self.stage:
            raise AgentNetConfigurationError("TrainingCandidate stage cannot be empty")
        object.__setattr__(self, "changes", tuple(str(change) for change in self.changes))


class ExplicitCandidates:
    """Use explicit network candidates through the same simple training workflow."""

    def __init__(self, candidates: Iterable[Module]) -> None:
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise AgentNetConfigurationError("ExplicitCandidates requires at least one network")
        if any(not isinstance(candidate, Module) for candidate in self.candidates):
            raise AgentNetConfigurationError(
                "ExplicitCandidates entries must be Module instances"
            )

    def generate(self, seed: Module) -> tuple[TrainingCandidate, ...]:
        candidates = self.candidates
        if all(candidate is not seed for candidate in candidates):
            candidates = (seed, *candidates)
        return tuple(
            TrainingCandidate(candidate, "explicit", ("explicit_candidate",))
            for candidate in candidates
        )


class AutoOptimizer:
    """Generate a small, staged set of prompt, topology, and policy candidates."""

    def __init__(
        self,
        *,
        optimizer_llm: Any | None = None,
        max_prompt_candidates: int = 2,
        optimize_prompts: bool = True,
        optimize_topology: bool = True,
        optimize_policies: bool = True,
    ) -> None:
        if max_prompt_candidates < 0:
            raise AgentNetConfigurationError(
                "AutoOptimizer max_prompt_candidates cannot be negative"
            )
        self.optimizer_llm = optimizer_llm
        self.max_prompt_candidates = max_prompt_candidates
        self.optimize_prompts = bool(optimize_prompts)
        self.optimize_topology = bool(optimize_topology)
        self.optimize_policies = bool(optimize_policies)

    def proposal_llm(self, net: Module) -> Any | None:
        if self.optimizer_llm is not None:
            if not hasattr(self.optimizer_llm, "complete"):
                raise AgentNetConfigurationError(
                    "optimizer_llm must provide an async complete method"
                )
            return self.optimizer_llm
        for agent in iter_react_agents(net):
            for llm in agent.llms:
                if hasattr(llm, "complete") and not isinstance(llm, FakeLLM):
                    return llm
        return None

    async def prompt_candidates(
        self,
        net: Module,
        *,
        failure_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[TrainingCandidate, ...]:
        if not self.optimize_prompts or self.max_prompt_candidates == 0:
            return ()
        proposal_llm = self.proposal_llm(net)
        agents = tuple(iter_react_agents(net))
        if proposal_llm is None or not agents:
            return ()
        response = await proposal_llm.complete(
            ChatRequest(
                model=str(proposal_llm.name),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Improve agent instructions using the failure examples. "
                            "Return JSON: {\"prompts\":[{\"target\":\"agent-name\","
                            "\"instructions\":\"...\"}]} and no executable code."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "agents": [
                                    {
                                        "instructions": agent.instructions,
                                        "name": agent.name,
                                    }
                                    for agent in agents
                                ],
                                "failures": list(failure_examples[:5]),
                            },
                            default=str,
                            sort_keys=True,
                        ),
                    },
                ],
            )
        )
        proposals = _parse_prompt_proposals(response.content, agents)
        candidates: list[TrainingCandidate] = []
        for target, instructions in proposals[: self.max_prompt_candidates]:
            replacement = _replace_agent_instructions(net, target, instructions)
            if replacement is not None:
                candidates.append(
                    TrainingCandidate(
                        replacement,
                        "prompt",
                        (f"prompt:{target}",),
                    )
                )
        return tuple(candidates)

    async def topology_candidates(
        self,
        net: Module,
        *,
        failure_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[TrainingCandidate, ...]:
        if not self.optimize_topology:
            return ()
        proposal_llm = self.proposal_llm(net)
        agents = tuple(iter_react_agents(net))
        if proposal_llm is None or not agents:
            return ()
        base_agent = agents[0]
        response = await proposal_llm.complete(
            ChatRequest(
                model=str(proposal_llm.name),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Design one bounded specialist branch for the failed cases. "
                            "Return JSON with string fields specialist and reducer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"failures": list(failure_examples[:5])},
                            default=str,
                            sort_keys=True,
                        ),
                    },
                ],
            )
        )
        specialist_prompt, reducer_prompt = _parse_topology_proposal(response.content)
        specialist = ReActAgent(
            f"{base_agent.name}_specialist",
            instructions=specialist_prompt,
            llms=base_agent.llms,
            tools=(),
            retry_policy=base_agent.retry_policy,
            max_steps=base_agent.max_steps,
            metadata={"generated_by": "AutoOptimizer"},
        )
        reducer = ReActAgent(
            f"{base_agent.name}_reducer",
            instructions=reducer_prompt,
            llms=base_agent.llms,
            tools=(),
            retry_policy=base_agent.retry_policy,
            interface=base_agent.interface,
            max_steps=base_agent.max_steps,
            metadata={"generated_by": "AutoOptimizer"},
        )
        return (
            TrainingCandidate(
                Parallel(
                    net,
                    specialist,
                    reducer=reducer,
                    name=f"{net.name}_ensemble",
                ),
                "topology",
                ("topology:add_specialist_branch",),
            ),
        )

    def policy_candidates(self, net: Module) -> tuple[TrainingCandidate, ...]:
        if not self.optimize_policies:
            return ()
        agents = tuple(iter_react_agents(net))
        candidates: list[TrainingCandidate] = []
        for agent in agents:
            if len(agent.llms) > 2:
                reordered = (agent.llms[0], *reversed(agent.llms[1:]))
                replacement = _replace_agent_policy(net, agent.name, llms=reordered)
                if replacement is not None:
                    candidates.append(
                        TrainingCandidate(
                            replacement,
                            "policy",
                            (f"fallback_order:{agent.name}",),
                        )
                    )
                break
        if agents:
            agent = agents[0]
            retry_policy = RetryPolicy(
                transport_retries=1,
                quality_retries=1,
                backoff="none",
                max_total_attempts=3,
            )
            if agent.retry_policy != retry_policy:
                replacement = _replace_agent_policy(
                    net,
                    agent.name,
                    retry_policy=retry_policy,
                )
                if replacement is not None:
                    candidates.append(
                        TrainingCandidate(
                            replacement,
                            "policy",
                            (f"retry_policy:{agent.name}",),
                        )
                    )
        return tuple(candidates)


def iter_react_agents(module: Module) -> Iterable[ReActAgent]:
    """Yield ReAct agents in stable execution-structure order."""

    if isinstance(module, ReActAgent):
        yield module
    elif isinstance(module, Sequential | Parallel):
        for child in module.modules:
            yield from iter_react_agents(child)
        if isinstance(module, Parallel) and module.reducer is not None:
            yield from iter_react_agents(module.reducer)
    elif isinstance(module, Router):
        yield from iter_react_agents(module.router)
        for child in module.routes.values():
            yield from iter_react_agents(child)
        if module.fallback is not None:
            yield from iter_react_agents(module.fallback)
    elif isinstance(module, Reducer):
        yield from iter_react_agents(module.reducer)
    elif isinstance(module, DAG):
        for child in module.nodes.values():
            yield from iter_react_agents(child)


def module_complexity(module: Module) -> tuple[int, int, int]:
    """Return recursive node, branch, and depth counts for deterministic ties."""

    if isinstance(module, ReActAgent):
        return (1, 0, 1)
    if isinstance(module, Sequential):
        metrics = [module_complexity(child) for child in module.modules]
        return (
            sum(metric[0] for metric in metrics),
            sum(metric[1] for metric in metrics),
            sum(metric[2] for metric in metrics),
        )
    if isinstance(module, Parallel):
        children = [module_complexity(child) for child in module.modules]
        if module.reducer is not None:
            children.append(module_complexity(module.reducer))
        return (
            sum(metric[0] for metric in children),
            len(module.modules) + sum(metric[1] for metric in children),
            1 + max((metric[2] for metric in children), default=0),
        )
    if isinstance(module, Router):
        children = [module_complexity(module.router)]
        children.extend(module_complexity(child) for child in module.routes.values())
        if module.fallback is not None:
            children.append(module_complexity(module.fallback))
        return (
            sum(metric[0] for metric in children),
            len(module.routes) + sum(metric[1] for metric in children),
            1 + max(metric[2] for metric in children),
        )
    if isinstance(module, Reducer):
        return module_complexity(module.reducer)
    if isinstance(module, DAG):
        metrics = [module_complexity(child) for child in module.nodes.values()]
        branches = sum(max(0, len(targets) - 1) for targets in module.edges.values())
        return (
            sum(metric[0] for metric in metrics),
            branches + sum(metric[1] for metric in metrics),
            _dag_depth(module),
        )
    return (1, 0, 1)


def estimate_llm_calls(module: Module, example_count: int) -> int:
    return sum(_agent_call_bound(agent) for agent in iter_react_agents(module)) * example_count


def _agent_call_bound(agent: ReActAgent) -> int:
    policy = agent.retry_policy
    if not isinstance(policy, RetryPolicy):
        return 1
    attempts = (policy.transport_retries + 1) * (policy.quality_retries + 1)
    if policy.max_total_attempts is not None:
        attempts = min(attempts, policy.max_total_attempts)
    return attempts * max(1, len(agent.llms))


def _parse_prompt_proposals(
    content: str,
    agents: Sequence[ReActAgent],
) -> list[tuple[str, str]]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None
    proposals: list[tuple[str, str]] = []
    if isinstance(payload, Mapping):
        raw_proposals = payload.get("prompts", ())
        if isinstance(raw_proposals, Sequence) and not isinstance(raw_proposals, str):
            known = {agent.name for agent in agents}
            for proposal in raw_proposals:
                if not isinstance(proposal, Mapping):
                    continue
                target = str(proposal.get("target", ""))
                instructions = str(proposal.get("instructions", "")).strip()
                if target in known and instructions:
                    proposals.append((target, instructions[:4000]))
    if not proposals and content.strip() and agents:
        proposals.append((agents[0].name, content.strip()[:4000]))
    return proposals


def _parse_topology_proposal(content: str) -> tuple[str, str]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, Mapping):
        specialist = str(payload.get("specialist", "")).strip()
        reducer = str(payload.get("reducer", "")).strip()
    else:
        specialist = content.strip()
        reducer = "Choose the most accurate branch answer and return only the final answer."
    return (
        specialist[:4000]
        or "Independently solve cases the primary agent may miss and return only the answer.",
        reducer[:4000]
        or "Choose the most accurate branch answer and return only the final answer.",
    )


def _replace_agent_instructions(
    module: Module,
    target: str,
    instructions: str,
) -> Module | None:
    return _replace_agent(module, target, instructions=instructions)


def _replace_agent_policy(
    module: Module,
    target: str,
    *,
    llms: Sequence[Any] | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Module | None:
    return _replace_agent(
        module,
        target,
        llms=llms,
        retry_policy=retry_policy,
    )


def _replace_agent(
    module: Module,
    target: str,
    *,
    instructions: str | None = None,
    llms: Sequence[Any] | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Module | None:
    if isinstance(module, ReActAgent):
        if module.name != target:
            return None
        return ReActAgent(
            module.name,
            instructions=module.instructions if instructions is None else instructions,
            llms=module.llms if llms is None else llms,
            tools=module.tools,
            retry_policy=(
                module.retry_policy if retry_policy is None else retry_policy
            ),
            input_interface=module.input_interface,
            interface=module.interface,
            max_steps=module.max_steps,
            metadata=module.metadata,
        )
    if isinstance(module, Sequential):
        return _replace_in_sequence(module, target, instructions, llms, retry_policy)
    if isinstance(module, Parallel):
        children, changed = _replace_children(
            module.modules,
            target,
            instructions,
            llms,
            retry_policy,
        )
        reducer = module.reducer
        if reducer is not None:
            replacement = _replace_agent(
                reducer,
                target,
                instructions=instructions,
                llms=llms,
                retry_policy=retry_policy,
            )
            if replacement is not None:
                reducer = replacement
                changed = True
        return Parallel(*children, reducer=reducer, name=module.name) if changed else None
    if isinstance(module, Router):
        return _replace_in_router(module, target, instructions, llms, retry_policy)
    if isinstance(module, Reducer):
        replacement = _replace_agent(
            module.reducer,
            target,
            instructions=instructions,
            llms=llms,
            retry_policy=retry_policy,
        )
        return Reducer(replacement, name=module.name) if replacement is not None else None
    if isinstance(module, DAG):
        nodes = dict(module.nodes)
        changed = False
        for name, child in module.nodes.items():
            replacement = _replace_agent(
                child,
                target,
                instructions=instructions,
                llms=llms,
                retry_policy=retry_policy,
            )
            if replacement is not None:
                nodes[name] = replacement
                changed = True
        return DAG(nodes=nodes, edges=module.edges, name=module.name) if changed else None
    return None


def _replace_in_sequence(
    module: Sequential,
    target: str,
    instructions: str | None,
    llms: Sequence[Any] | None,
    retry_policy: RetryPolicy | None,
) -> Module | None:
    children, changed = _replace_children(
        module.modules,
        target,
        instructions,
        llms,
        retry_policy,
    )
    return Sequential(*children, name=module.name) if changed else None


def _replace_children(
    children: Sequence[Module],
    target: str,
    instructions: str | None,
    llms: Sequence[Any] | None,
    retry_policy: RetryPolicy | None,
) -> tuple[list[Module], bool]:
    replaced_children: list[Module] = []
    changed = False
    for child in children:
        replacement = _replace_agent(
            child,
            target,
            instructions=instructions,
            llms=llms,
            retry_policy=retry_policy,
        )
        replaced_children.append(child if replacement is None else replacement)
        changed = changed or replacement is not None
    return replaced_children, changed


def _replace_in_router(
    module: Router,
    target: str,
    instructions: str | None,
    llms: Sequence[Any] | None,
    retry_policy: RetryPolicy | None,
) -> Module | None:
    changed = False
    router = _replace_agent(
        module.router,
        target,
        instructions=instructions,
        llms=llms,
        retry_policy=retry_policy,
    )
    if router is None:
        router = module.router
    else:
        changed = True
    routes: dict[str, Module] = {}
    for name, child in module.routes.items():
        replacement = _replace_agent(
            child,
            target,
            instructions=instructions,
            llms=llms,
            retry_policy=retry_policy,
        )
        routes[name] = child if replacement is None else replacement
        changed = changed or replacement is not None
    fallback = module.fallback
    if fallback is not None:
        replacement = _replace_agent(
            fallback,
            target,
            instructions=instructions,
            llms=llms,
            retry_policy=retry_policy,
        )
        if replacement is not None:
            fallback = replacement
            changed = True
    if not changed:
        return None
    return Router(router=router, routes=routes, fallback=fallback, name=module.name)


def _dag_depth(module: DAG) -> int:
    depths: dict[str, int] = {}

    def visit(node: str) -> int:
        if node in depths:
            return depths[node]
        targets = module.edges.get(node, ())
        depth = 1 if not targets else 1 + max(visit(target) for target in targets)
        depths[node] = depth
        return depth

    entries = tuple(name for name, predecessors in module.predecessors.items() if not predecessors)
    return max((visit(name) for name in entries), default=1)
