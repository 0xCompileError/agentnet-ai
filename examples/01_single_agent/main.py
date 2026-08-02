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
    context = an.RunContext(run_id="example-single-agent")
    agent = an.ReActAgent(
        "planner",
        instructions="Return one concise next step.",
        llms=[
            an.FakeLLM(
                ["Inventory the current workflow before changing it."],
                name="strong",
            )
        ],
    )

    output = an.run(agent, "Prepare a migration plan.", context=context)
    agent_state = context.graph_state.get_agent_state("planner")
    emit(
        "single_agent",
        {
            "output": output,
            "reasoning_steps": agent_state.step,
        },
    )


if __name__ == "__main__":
    main()
