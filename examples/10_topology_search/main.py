# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


class Classifier(an.Module):
    def __init__(self, name: str, labels: dict[str, str]) -> None:
        super().__init__(name)
        self.labels = labels

    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return self.labels.get(str(input), "unknown")


def main() -> None:
    inputs = ["invoice", "timeout", "refund", "crash", "charged twice"]
    labels = ["billing", "technical", "billing", "technical", "billing"]
    seed = Classifier("seed", {value: "unknown" for value in inputs})
    replacement = Classifier(
        "seed_v2",
        dict(zip(inputs, labels, strict=True)),
    )
    search_space = an.TopologySearchSpace(
        allowed_mutations=["node_replacement"],
        max_nodes=1,
        max_trials=2,
        replacement_candidates=[replacement],
    )
    optimizer = an.TopologyOptimizer(search_space=search_space)
    trained = an.train(
        seed,
        inputs,
        labels,
        optimize=optimizer,
    )

    emit(
        "topology_search",
        {
            "best_net": trained.name,
            "best_score": trained.training.best_score,
            "topology_trials": len(trained.training.trials),
        },
    )


if __name__ == "__main__":
    main()
