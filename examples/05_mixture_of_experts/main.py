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
    experts = {
        "finance": an.ReActAgent(
            "finance_expert",
            instructions="Assess cost and budget impact.",
            llms=[an.FakeLLM(["Finance expert: cap pilot spend first."], name="finance")],
        ),
        "security": an.ReActAgent(
            "security_expert",
            instructions="Assess security impact.",
            llms=[an.FakeLLM(["Security expert: approve data boundary first."], name="security")],
        ),
    }
    mixture = an.Router(
        router=an.ReActAgent(
            "expert_selector",
            instructions="Choose finance or security.",
            llms=[an.FakeLLM(["security"], name="selector")],
        ),
        routes=experts,
        fallback=experts["finance"],
        name="expert_mixture",
    )

    output = an.run(mixture, "Can this workflow handle regulated records?")
    emit(
        "mixture_of_experts",
        {
            "expert_count": len(experts),
            "output": output,
            "selected_expert": "security",
        },
    )


if __name__ == "__main__":
    main()
