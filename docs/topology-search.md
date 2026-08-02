# Topology Search Guide

Topology search explores bounded architecture changes around a seed `Module`.
It is an optimizer workflow, not runtime behavior. `TopologySearchSpace`
declares limits and explicit candidate modules, and `TopologyOptimizer.search`
generates and scores candidates.

## Search Space

Bound the search before generating candidates. `max_trials` is the first safety
limit to set when experimenting.

```python
import agentnet as an

seed = an.ReActAgent("planner", llms=[an.FakeLLM(["seed"], name="strong")])
branch = an.ReActAgent("critic", llms=[an.FakeLLM(["branch"], name="strong")])
replacement = an.ReActAgent("planner_v2", llms=[an.FakeLLM(["replacement"], name="strong")])

search_space = an.TopologySearchSpace(
    max_nodes=4,
    max_branches=2,
    max_depth=3,
    max_trials=5,
    branch_candidates=[branch],
    replacement_candidates=[replacement],
)
```

The mutation engine supports branch insertion, branch removal, router insertion,
reducer insertion, and node replacement. Candidates are built from modules you
provide in process; AgentNet does not import arbitrary mutation code.

## Train Through Topology Search

Pass a bounded topology optimizer directly to the simple training API. AgentNet
executes each live candidate module and owns dataset scoring, validation, and
selection:

```python
import agentnet as an

optimizer = an.TopologyOptimizer(search_space=search_space)
trained = an.train(
    seed,
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    optimize=optimizer,
)

print(trained.training.summary())
for trial in trained.training.trials:
    print(trial.to_dict())
```

## Constraints

Topology search can validate candidates with training constraints before
scoring:

```python
import agentnet as an

constraint = an.GraphConstraint(
    an.TopologyConstraint(max_nodes=4, max_branches=2, max_depth=3)
)
trained = an.train(
    seed,
    x_train,
    y_train,
    optimize=an.TopologyOptimizer(
        search_space=search_space,
        constraints=[constraint],
    ),
)
```

Use constraints for hard architecture limits that must apply across optimizers,
artifacts, and enterprise policy checks.

## Trace Topology Trials

Topology results can be recorded into runtime context metadata and normalized
with the tracing system:

```python
import agentnet as an

print(trained.training.best_trial.changes)
print(trained.training.best_trial.complexity)
```
