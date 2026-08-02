# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def create_run(self, **payload: Any) -> None:
        self.runs.append(payload)


def main() -> None:
    context = an.RunContext(run_id="example-langsmith")
    tracer = an.InMemoryTracer()
    agent = an.ReActAgent(
        "planner",
        llms=[an.FakeLLM(["Traced output."], name="strong")],
    )

    output = an.run(agent, "Trace this run.", context=context, tracer=tracer)
    trace = an.trace_from_context(context)
    client = FakeLangSmithClient()
    exported = an.LangSmithExporter(
        client=client,
        project_name="agentnet-examples",
    ).export(trace)

    emit(
        "langsmith",
        {
            "exported_runs": len(exported),
            "output": output,
            "span_names": [run["name"] for run in client.runs],
        },
    )


if __name__ == "__main__":
    main()
