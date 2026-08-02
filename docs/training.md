# Training Guide

AgentNet's default training path accepts ordinary inputs and expected outputs:

```python
import agentnet as an

trained = an.train(net, x_train, y_train)
answer = trained.run(new_input)
```

`an.train` is evaluation-driven: it measures bounded prompt, topology, and
policy candidates and returns a separate runnable `FittedAgentNet`. It never
mutates the network passed by the caller and does not perform gradient descent.

## Defaults

The simple call:

- converts `X` and `y` into training examples;
- infers expected-output scoring, trimming and case-folding string labels;
- reserves a deterministic 20% validation set when at least five examples are
  available;
- evaluates the unchanged baseline first;
- uses a bounded automatic optimizer with at most ten trials and 200 estimated
  LLM calls;
- ranks by validation score, training score, structural simplicity, and stable
  trial order; and
- records sanitized provenance under `trained.training`.

For smaller datasets, AgentNet uses training-score selection and records a
warning instead of silently discarding scarce examples.

```python
print(trained.training.summary())
print(trained.training.to_dict())
trained.save("trained.agentnet")
```

Serialized reports contain scores, descriptors, safe change identifiers, and
error types. They do not contain inputs, expected outputs, generated prompts,
exception messages, credentials, or live clients.

## Progressive Controls

Supply only the controls the application needs:

```python
trained = an.train(
    net,
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    optimize="topology",
    objective=custom_objective,
    optimizer_llm=optimization_model,
    budget=an.Budget(max_trials=20, max_llm_calls=1_000),
    random_state=42,
)
```

Accepted strategies are `"auto"`, `"prompt"`, `"topology"`, `"policy"`, and
`"none"`. Advanced callers can pass `AutoOptimizer`, `TopologyOptimizer`, or
`ExplicitCandidates` directly. `atrain` provides the equivalent async API.

Automatic semantic proposals use the first suitable live LLM already attached
to the network, unless `optimizer_llm` is supplied. Generated agents inherit an
existing model policy, cannot gain tool permissions, and are limited to
declarative built-in graph modules.

## Dataset API

`Dataset` is an immutable collection of `TrainingExample` objects. Metadata is
validated to avoid secret-like keys.

```python
import agentnet as an

dataset = an.Dataset(
    [
        an.TrainingExample(
            "Summarize the incident.",
            expected_output="short summary",
            id="case-1",
        ),
    ],
    name="incident-summary",
)
```

## Objective

Use built-in objectives for schema checks, exact matches, judge callbacks, human
feedback, unit tests, cost, latency, and tool efficiency. Use `CustomObjective`
when scoring needs dataset context supplied by the trainer.

```python
import agentnet as an


def matches_expected(output: object, context: object | None) -> dict[str, object]:
    expected = context["expected_output"] if isinstance(context, dict) else None
    score = 1.0 if output == expected else 0.0
    return {"score": score, "passed": score == 1.0}


objective = (
    an.CustomObjective("expected_match", matches_expected, threshold=1.0)
    + an.UnitTestObjective(lambda output: isinstance(output, str), name="is_text")
)
```

Judge and custom objectives run only explicit in-process callables supplied by
application code. AgentNet does not deserialize objective code from artifacts.

## Advanced Trainer

`Trainer` remains available for applications that need direct control over
candidate evaluation, checkpoints, and per-example lifecycle events. New code
should prefer `an.train` unless it specifically needs those internals.

```python
import agentnet as an

baseline = an.ReActAgent(
    "summarizer",
    instructions="Write a short summary.",
    llms=[an.FakeLLM(["short summary"], name="strong")],
)

candidate = an.ReActAgent(
    "summarizer_candidate",
    instructions="Write a short, factual summary.",
    llms=[an.FakeLLM(["short summary"], name="strong")],
)

trainer = an.Trainer(
    net=baseline,
    objective=an.CustomObjective("expected_match", matches_expected, threshold=1.0),
    budget=an.Budget(max_epochs=3, max_examples=10, max_trials=3),
)

result = trainer.fit(dataset, epochs=1, candidates=[baseline, candidate])

print(result.score)
print(result.history.to_dict())
for candidate_result in result.candidate_results:
    print(candidate_result.candidate.name, candidate_result.score)
```

Candidate results remain in evaluation order and contain each live candidate,
its objective result, independent history, and checkpoint. The legacy `net`,
`score`, `objective_result`, and `history` attributes still refer to the selected
candidate. Exact ties keep selecting the first candidate while `is_tied`,
`tied_candidates`, and `metadata["selection_policy"]` make that outcome explicit.

Pass a callback to observe prompt-free lifecycle events. Callback errors are
allowed to propagate so monitoring failures stop training immediately:

```python
def show_progress(event: an.TrainingProgressEvent) -> None:
    if event.event_type.endswith(".completed"):
        print(event.to_dict())


result = trainer.fit(dataset, progress_callback=show_progress)
```

## Checkpoints And History

Each fit result includes descriptor-only checkpoints and training history. Save
history with the artifact when you want downstream validation or package exports
to include training provenance:

```python
import agentnet as an

an.save(
    result.net,
    "summarizer.agentnet",
    name="summarizer",
    training_history=result.history,
)
```

Training descriptors contain metrics, scores, candidate descriptors, and
metadata. They do not contain live networks, live clients, or executable patches.

## Runnable 10-Example Training Script

The repository includes a deterministic training example that builds a
10-example support-triage dataset, evaluates two `ReActAgent` candidates, and
selects the better candidate:

```bash
uv run python examples/12_training_10_examples/main.py
```

Use OpenAI for real LLM calls by opting in explicitly. This evaluates two
candidates over 10 examples, so it makes 20 chat-completion calls:

```bash
AGENTNET_TRAINING_LLM=openai \
uv run --env-file .env python examples/12_training_10_examples/main.py
```

The `.env` file must define `OPENAI_API_KEY` and `OPENAI_MODEL`. AgentNet does
not load `.env` files or depend on a dotenv package; `uv` injects the file for
this command. Progress is written to stderr and the final JSON stays on stdout.
