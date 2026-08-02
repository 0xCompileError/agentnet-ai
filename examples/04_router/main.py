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
    router = an.Router(
        router=an.ReActAgent(
            "classifier",
            instructions="Return billing or technical.",
            llms=[an.FakeLLM(["technical"], name="router-model")],
        ),
        routes={
            "billing": an.ReActAgent(
                "billing",
                llms=[an.FakeLLM(["Ask for invoice ID."], name="billing")],
            ),
            "technical": an.ReActAgent(
                "technical",
                llms=[an.FakeLLM(["Collect logs and reproduction steps."], name="technical")],
            ),
        },
        name="support_router",
    )

    output = an.run(router, "The deployment job fails after checkout.")
    emit(
        "router",
        {
            "output": output,
            "selected_route": "technical",
        },
    )


if __name__ == "__main__":
    main()
