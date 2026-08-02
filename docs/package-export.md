# Package Export Guide

Package export turns a validated `.agentnet` artifact into a standard Python
package. The generated package embeds the artifact and exposes thin loader
helpers that delegate to AgentNet.

## Export From Python

```python
import agentnet as an

artifact = "decision_net.agentnet"

result = an.export_package(
    artifact,
    "dist/decision-net",
    package_name="decision-net",
    description="Decision support AgentNet package",
    overwrite=True,
)

print(result.pyproject_path)
print(result.loader_path)
```

`export_package` validates the source artifact before generating the project.
The output uses a `src/` layout, Hatchling metadata, `py.typed`, and a loader
module.

## Export From CLI

Use `agentnet export` in descriptor workflows:

```bash
agentnet export decision_net.agentnet --package decision-net --output dist/decision-net
```

Add `--module-name`, `--version`, `--description`, `--overwrite`, or `--json`
when automation needs deterministic package metadata or machine-readable output.

## Load A Generated Package

Consumers import the generated module and provide explicit dependency injection:

```python
import agentnet as an
import decision_net

net = decision_net.load(
    llms={"strong": an.FakeLLM(["ok"], name="strong")},
    tools={"search_docs": lambda query: [query]},
)

print(an.run(net, "Assess the migration."))
```

Generated packages do not contain provider clients, tool implementations,
credentials, or remote scheduler clients. They contain descriptors and loader
code only.

## Validate Before Running

The generated module also exposes validation:

```python
validation = decision_net.validate(
    llms={"strong": an.FakeLLM(name="strong")},
    tools={"search_docs": lambda query: [query]},
)

if not validation.passed:
    raise RuntimeError(validation.failures)
```

Use validation in CI and deployment checks before publishing an exported
package.
