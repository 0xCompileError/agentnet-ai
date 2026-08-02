# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import CallableModule, emit


def main() -> None:
    reducer = CallableModule(
        "synthesizer",
        lambda outputs: {
            "branches": list(outputs),
            "summary": "Latency and support risk both need owners.",
        },
    )
    research = an.Parallel(
        an.ReActAgent(
            "latency_researcher",
            instructions="Research latency risks.",
            llms=[an.FakeLLM(["Latency: measure p95 before rollout."], name="latency")],
        ),
        an.ReActAgent(
            "support_researcher",
            instructions="Research support risks.",
            llms=[an.FakeLLM(["Support: update runbooks first."], name="support")],
        ),
        reducer=reducer,
        name="research_parallel",
    )

    output = an.run(research, "Assess launch readiness.")
    emit("parallel_research_pipeline", output)


if __name__ == "__main__":
    main()
