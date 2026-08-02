# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


def main() -> None:
    pipeline = an.Sequential(
        an.ReActAgent(
            "planner",
            instructions="Plan the response.",
            llms=[an.FakeLLM(["Plan: compare risk, cost, and rollout."], name="planner")],
        ),
        an.ReActAgent(
            "critic",
            instructions="Identify gaps.",
            llms=[an.FakeLLM(["Gap: confirm rollback ownership."], name="critic")],
        ),
        an.ReActAgent(
            "writer",
            instructions="Write the final answer.",
            llms=[an.FakeLLM(["Recommendation: migrate in phases."], name="writer")],
        ),
        name="migration_pipeline",
    )

    output = an.run(pipeline, "Should we migrate the nightly job?")
    graph = an.compile_graph(pipeline)
    emit(
        "sequential_pipeline",
        {
            "node_count": len(graph.nodes),
            "output": output,
        },
    )


if __name__ == "__main__":
    main()
