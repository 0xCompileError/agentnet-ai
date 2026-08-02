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
    context = an.RunContext(run_id="example-distributed")
    scheduler = an.ThreadPoolScheduler(max_workers=1)
    try:
        output = an.run(
            CallableModule("uppercase", lambda value: str(value).upper()),
            "scheduled work",
            context=context,
            scheduler=scheduler,
        )
    finally:
        scheduler.shutdown()

    emit(
        "distributed_execution",
        {
            "output": output,
            "scheduler_events": len(context.metadata["scheduler_events"]),
        },
    )


if __name__ == "__main__":
    main()
