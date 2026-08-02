# Quickstart

This guide builds a small AgentNet network, runs it with a deterministic local
model, saves a descriptor-only `.agentnet` artifact, and shows the matching CLI
workflow. The examples avoid network calls and credentials.

## Install For Development

```bash
uv sync --dev
uv run pytest
```

Create a project scaffold with the CLI when starting a new application:

```bash
agentnet init support-net
```

The scaffold contains a config file, a minimal package, an example agent, and
JSON evaluation cases. Use `--force` only when overwriting scaffold files is
intentional.

## Run One Agent

`FakeLLM` is useful for tests, examples, and local documentation because it
returns queued responses without a provider account.

```python
import agentnet as an

planner = an.ReActAgent(
    "planner",
    instructions="Return a concise migration plan.",
    llms=[an.FakeLLM(responses=["Inventory dependencies, test, then cut over."], name="strong")],
)

answer = an.run(planner, "Move the reporting job to the new warehouse.")
print(answer)
```

Production code usually replaces `FakeLLM` with an injected backend such as
`LiteLLM`, `OpenAI`, `Anthropic`, `Bedrock`, `VertexAI`, or another object that
implements the LLM protocol.

## Compose A Network

Graph containers compose agents while preserving runtime context:

```python
import agentnet as an

net = an.Sequential(
    an.ReActAgent(
        "planner",
        instructions="Break the request into steps.",
        llms=[an.FakeLLM(["Plan the analysis."], name="planner-model")],
    ),
    an.ReActAgent(
        "writer",
        instructions="Write the final response.",
        llms=[an.FakeLLM(["Final response."], name="writer-model")],
    ),
    name="decision_net",
)

print(an.run(net, "Should we move the batch job?"))
```

Use `Parallel`, `Router`, `Reducer`, `DAG`, or `build_shape` when the graph needs
branching, routing, fan-in, named dependencies, or shape-driven construction.

## Save And Load

Saving creates a descriptor-only `.agentnet` artifact. The artifact stores graph
shape, prompts, schemas, model aliases, tool descriptors, MCP descriptors, and
training metadata. It does not store live clients, callables, or secrets.

```python
import agentnet as an

net = an.ReActAgent(
    "planner",
    instructions="Plan clearly.",
    llms=["strong"],
)

an.save(net, "planner.agentnet", name="planner")

loaded = an.load(
    "planner.agentnet",
    llms={"strong": an.FakeLLM(responses=["loaded"], name="strong")},
)

print(an.run(loaded, "Use the saved artifact."))
```

The same artifact can be inspected and exported from the CLI:

```bash
agentnet inspect planner.agentnet
agentnet export planner.agentnet --package planner-net --output dist/planner-net
```

## Validate Locally

Run the same validation commands used by the repository workflow:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```
