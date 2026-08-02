import os
import threading
from typing import Any

import pytest

import agentnet as an


class EchoModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return input


class ThreadNameModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del input, context
        return threading.current_thread().name


class ProcessInfoModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return {"input": input, "pid": os.getpid()}


class FlakyModule(an.Module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.attempts = 0

    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("temporary worker failure")
        return input


class FakeRemoteClient:
    def __init__(self, label: str) -> None:
        self.label = label
        self.submissions: list[dict[str, Any]] = []

    def submit(
        self,
        module: an.Module,
        input: object,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.submissions.append(
            {
                "context": context,
                "input": input,
                "module": module.name,
            }
        )
        context["metadata"]["remote_label"] = self.label
        return {
            "context": context,
            "output": f"{self.label}:{module.name}:{input}",
            "succeeded": True,
        }

    def gather(self, handles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return handles


def test_node_spec_serializes_descriptor_without_live_module() -> None:
    module = EchoModule("echo")

    spec = an.NodeSpec(module, metadata={"worker_pool": "local"})

    assert spec.name == "echo"
    assert spec.to_dict() == {
        "metadata": {"worker_pool": "local"},
        "module": {"name": "echo", "type": "EchoModule"},
        "name": "echo",
    }


def test_node_spec_rejects_secret_like_metadata() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="may serialize secrets"):
        an.NodeSpec(EchoModule("echo"), metadata={"api_token": "secret"})


@pytest.mark.anyio
async def test_local_scheduler_submit_gather_executes_and_traces() -> None:
    context = an.RunContext(run_id="run-local")
    scheduler = an.LocalScheduler()

    future = await scheduler.submit(EchoModule("echo"), "payload", context)
    result = (await scheduler.gather([future]))[0]

    assert result.output == "payload"
    assert result.succeeded is True
    assert result.node_name == "echo"
    assert context.metadata["scheduler_events"] == [
        {
            "node": "echo",
            "run_id": "run-local",
            "scheduler": "local",
            "type": "scheduler.submit",
        },
        {
            "attempt": 1,
            "node": "echo",
            "run_id": "run-local",
            "scheduler": "local",
            "type": "scheduler.started",
        },
        {
            "attempt": 1,
            "node": "echo",
            "run_id": "run-local",
            "scheduler": "local",
            "type": "scheduler.completed",
        },
    ]


def test_package_run_uses_scheduler_and_returns_output() -> None:
    context = an.RunContext(run_id="run-package")

    output = an.run(EchoModule("echo"), "payload", context=context, scheduler=an.LocalScheduler())

    assert output == "payload"
    assert context.metadata["scheduler_events"][0]["type"] == "scheduler.submit"


@pytest.mark.anyio
async def test_thread_pool_scheduler_runs_module_in_worker_thread() -> None:
    scheduler = an.ThreadPoolScheduler(max_workers=1)
    try:
        result = await scheduler.arun(ThreadNameModule("thread-name"), None)
    finally:
        scheduler.shutdown()

    assert result.output.startswith("agentnet-thread")


def test_process_pool_scheduler_runs_picklable_module_in_worker_process() -> None:
    scheduler = an.ProcessPoolScheduler(max_workers=1)
    try:
        result = scheduler.run(ProcessInfoModule("process-info"), "payload")
    finally:
        scheduler.shutdown()

    assert result.output["input"] == "payload"
    assert result.output["pid"] != os.getpid()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("scheduler_name", "label"),
    [
        ("RayScheduler", "ray"),
        ("CeleryScheduler", "celery"),
        ("TemporalScheduler", "temporal"),
    ],
)
async def test_remote_scheduler_adapters_use_injected_clients(
    scheduler_name: str,
    label: str,
) -> None:
    client = FakeRemoteClient(label)
    context = an.RunContext(run_id=f"run-{label}")
    scheduler_cls = getattr(an, scheduler_name)
    scheduler = scheduler_cls(client=client)

    result = await scheduler.arun(EchoModule("echo"), "payload", context)

    assert result.output == f"{label}:echo:payload"
    assert client.submissions[0]["module"] == "echo"
    assert context.metadata["remote_label"] == label
    assert context.metadata["scheduler_events"][0]["scheduler"] == label


@pytest.mark.anyio
async def test_scheduler_retries_failed_node_execution_and_records_metrics() -> None:
    context = an.RunContext(run_id="run-retry")
    module = FlakyModule("flaky")
    scheduler = an.LocalScheduler(
        retry_policy=an.RetryPolicy(
            transport_retries=1,
            quality_retries=0,
            backoff="none",
        )
    )

    result = await scheduler.arun(module, "payload", context)

    assert result.output == "payload"
    assert result.attempts == 2
    assert module.attempts == 2
    assert context.metadata["scheduler_retry_events"] == [
        {
            "attempt": 1,
            "delay_seconds": 0.0,
            "error_type": "TimeoutError",
            "next_attempt": 2,
            "node": "flaky",
            "reason": "transport",
            "run_id": "run-retry",
            "scheduler": "local",
            "type": "scheduler.retry.started",
        }
    ]
    assert context.metadata["scheduler_metrics"] == {
        "total_backoff_seconds": 0.0,
        "total_retries": 1,
        "transport_retries": 1,
    }


def test_scheduler_public_exports_are_available() -> None:
    assert an.Scheduler is not None
    assert an.NodeFuture is not None
    assert an.NodeResult is not None
    assert an.NodeSpec is not None
    assert an.LocalScheduler is not None
    assert an.ThreadPoolScheduler is not None
    assert an.ProcessPoolScheduler is not None
    assert an.RayScheduler is not None
    assert an.CeleryScheduler is not None
    assert an.TemporalScheduler is not None
