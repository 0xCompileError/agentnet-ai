# ruff: noqa: E402, I001
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        artifact = temp_dir / "decision_net.agentnet"
        package_dir = temp_dir / "decision_package"
        an.save(
            an.ReActAgent(
                "planner",
                instructions="Answer from the exported artifact.",
                llms=["strong"],
            ),
            artifact,
            name="decision_net",
        )

        export_result = an.export_package(
            artifact,
            package_dir,
            package_name="decision-net",
            description="Example exported AgentNet package",
        )
        sys.path.insert(0, str(package_dir / "src"))
        try:
            sys.modules.pop("decision_net", None)
            decision_net = importlib.import_module("decision_net")
            llm = an.FakeLLM(["Loaded from exported package."], name="strong")
            validation = decision_net.validate(llms={"strong": llm})
            loaded = decision_net.load(llms={"strong": llm})
            output = an.run(loaded, "Run exported package.")
        finally:
            sys.path.remove(str(package_dir / "src"))

    emit(
        "exported_package",
        {
            "module_name": export_result.module_name,
            "output": output,
            "validation_passed": validation.passed,
        },
    )


if __name__ == "__main__":
    main()
