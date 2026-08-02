"""Deterministic release-candidate benchmark for AgentNet."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_agentnet() -> Any:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import agentnet as an

    return an


def run_benchmark(iterations: int = 25) -> dict[str, object]:
    an = _load_agentnet()
    llm = an.FakeLLM(
        name="strong",
        responses=[f"ok-{index}" for index in range(iterations)],
    )
    agent = an.ReActAgent(
        "planner",
        instructions="Return the next deterministic benchmark response.",
        llms=[llm],
    )

    started = time.perf_counter()
    outputs: list[str] = []
    for index in range(iterations):
        context = an.RunContext(f"benchmark-{index}")
        outputs.append(str(an.run(agent, f"input-{index}", context=context)))
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "agent_runs": len(outputs),
        "benchmark": "release_candidate",
        "elapsed_ms": round(elapsed_ms, 3),
        "iterations": iterations,
        "ok": outputs == [f"ok-{index}" for index in range(iterations)],
    }


def main() -> None:
    print(json.dumps(run_benchmark(), sort_keys=True))


if __name__ == "__main__":
    main()
