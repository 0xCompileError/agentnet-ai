from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agentnet as an

ROOT = Path(__file__).resolve().parents[1]


class DictStorage:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(self, key: str, value: object) -> None:
        self.values[key] = value

    def get(self, key: str) -> object:
        return self.values[key]


class ListMemory:
    def __init__(self) -> None:
        self.items: list[object] = []

    def append(self, value: object) -> None:
        self.items.append(value)


def test_plugin_descriptor_round_trips_without_executable_payload() -> None:
    descriptor = an.PluginDescriptor(
        "offline-provider",
        an.PluginKind.PROVIDER,
        version="1.2.0",
        description="Offline provider plugin",
        metadata={"owner": "platform"},
    )

    payload = descriptor.to_dict()

    assert payload == {
        "description": "Offline provider plugin",
        "kind": "provider",
        "metadata": {"owner": "platform"},
        "name": "offline-provider",
        "version": "1.2.0",
    }
    assert an.PluginDescriptor.from_dict(payload) == descriptor
    assert "lambda" not in str(payload)


def test_plugin_descriptor_rejects_secret_like_metadata() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="may serialize secrets"):
        an.PluginDescriptor(
            "unsafe",
            an.PluginKind.PROVIDER,
            metadata={"api_token": "not allowed"},
        )


def test_plugin_manager_registers_and_creates_all_plugin_categories() -> None:
    manager = an.PluginManager()

    manager.register_provider(
        "fake-llm",
        lambda response="ready": an.FakeLLM([response], name="fake-llm"),
    )
    manager.register_optimizer("prompt", lambda: an.PromptOptimizer())
    manager.register_evaluator("exact", lambda expected: an.ExactMatchObjective(expected))
    manager.register_scheduler("local", lambda: an.LocalScheduler())
    manager.register_tracer("memory", lambda: an.InMemoryTracer())
    manager.register_storage("dict", DictStorage)
    manager.register_memory("list", ListMemory)

    provider = manager.providers.create("fake-llm", response="done")
    evaluator = manager.evaluators.create("exact", "done")
    scheduler = manager.schedulers.create("local")
    tracer = manager.tracers.create("memory")
    storage = manager.storage.create("dict")
    memory = manager.memory.create("list")

    assert isinstance(provider, an.FakeLLM)
    assert isinstance(manager.optimizers.create("prompt"), an.PromptOptimizer)
    assert evaluator.evaluate("done").passed is True
    assert isinstance(scheduler, an.Scheduler)
    assert isinstance(tracer, an.InMemoryTracer)
    assert isinstance(storage, DictStorage)
    assert isinstance(memory, ListMemory)
    assert manager.create(an.PluginKind.PROVIDER, "fake-llm").name == "fake-llm"

    payload = manager.to_dict()
    assert payload["providers"]["plugins"][0]["name"] == "fake-llm"
    assert payload["evaluators"]["plugins"][0]["kind"] == "evaluator"
    assert "lambda" not in str(payload)


def test_plugin_registries_validate_factory_results() -> None:
    provider_plugins = an.ProviderPluginRegistry()
    provider_plugins.register("bad-provider", lambda: object())

    evaluator_plugins = an.EvaluatorPluginRegistry()
    evaluator_plugins.register("bad-evaluator", lambda: object())

    scheduler_plugins = an.SchedulerPluginRegistry()
    scheduler_plugins.register("bad-scheduler", lambda: object())

    tracer_plugins = an.TracerPluginRegistry()
    tracer_plugins.register("bad-tracer", lambda: object())

    with pytest.raises(an.AgentNetConfigurationError, match="LLMBackend"):
        provider_plugins.create("bad-provider")
    with pytest.raises(an.AgentNetConfigurationError, match="Objective"):
        evaluator_plugins.create("bad-evaluator")
    with pytest.raises(an.AgentNetConfigurationError, match="Scheduler"):
        scheduler_plugins.create("bad-scheduler")
    with pytest.raises(an.AgentNetConfigurationError, match="tracer"):
        tracer_plugins.create("bad-tracer")


def test_descriptor_only_plugin_registries_do_not_create_plugins() -> None:
    registry = an.ProviderPluginRegistry.from_dict(
        {
            "plugins": [
                an.PluginDescriptor(
                    "fake-llm",
                    an.PluginKind.PROVIDER,
                    description="Descriptor only",
                ).to_dict()
            ]
        }
    )

    assert registry.names == ("fake-llm",)
    assert registry.to_dict()["plugins"][0]["name"] == "fake-llm"
    with pytest.raises(an.AgentNetConfigurationError, match="no registered factory"):
        registry.create("fake-llm")


def test_plugin_registry_rejects_duplicates_and_unknown_kinds() -> None:
    registry = an.StoragePluginRegistry()
    registry.register("dict", DictStorage)

    with pytest.raises(an.AgentNetConfigurationError, match="already registered"):
        registry.register("dict", DictStorage)

    manager = an.PluginManager()
    with pytest.raises(an.AgentNetConfigurationError, match="Unknown plugin kind"):
        manager.create("missing", "dict")


def test_plugin_public_exports_and_documentation_are_available() -> None:
    exported: list[Any] = [
        an.PluginDescriptor,
        an.PluginKind,
        an.PluginManager,
        an.ProviderPluginRegistry,
        an.OptimizerPluginRegistry,
        an.EvaluatorPluginRegistry,
        an.SchedulerPluginRegistry,
        an.TracerPluginRegistry,
        an.StoragePluginRegistry,
        an.MemoryPluginRegistry,
    ]

    assert all(value is not None for value in exported)

    docs = (ROOT / "docs" / "plugins.md").read_text()
    assert "PluginManager" in docs
    assert "ProviderPluginRegistry" in docs
    assert "StoragePluginRegistry" in docs
    assert "does not load executable code" in docs
