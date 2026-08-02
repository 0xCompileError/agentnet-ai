"""Descriptor-safe plugin registries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError
from agentnet.evaluation import Objective
from agentnet.llms import LLMBackend
from agentnet.mcp._security import validate_safe_metadata
from agentnet.runtime import Scheduler


class PluginKind(StrEnum):
    """Supported plugin categories."""

    PROVIDER = "provider"
    OPTIMIZER = "optimizer"
    EVALUATOR = "evaluator"
    SCHEDULER = "scheduler"
    TRACER = "tracer"
    STORAGE = "storage"
    MEMORY = "memory"


@dataclass(frozen=True, slots=True, init=False)
class PluginDescriptor:
    """Serializable plugin descriptor without executable code."""

    name: str
    kind: PluginKind
    version: str
    description: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        kind: PluginKind | str,
        *,
        version: str = "1",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("Plugin name cannot be empty")
        if not version:
            raise AgentNetConfigurationError("Plugin version cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="PluginDescriptor")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", _coerce_plugin_kind(kind))
        object.__setattr__(self, "version", str(version))
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "kind": self.kind.value,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, descriptor: Mapping[str, Any]) -> Self:
        return cls(
            str(descriptor["name"]),
            str(descriptor["kind"]),
            version=str(descriptor.get("version", "1")),
            description=(
                None
                if descriptor.get("description") is None
                else str(descriptor["description"])
            ),
            metadata=dict(descriptor.get("metadata", {})),
        )


PluginFactory = Callable[..., Any]
PluginValidator = Callable[[str, Any], None]


class PluginRegistry:
    """Registry for explicit plugin factories and descriptor-only records."""

    kind: PluginKind

    def __init__(
        self,
        kind: PluginKind | str,
        *,
        factories: Mapping[str, PluginFactory] | None = None,
        validator: PluginValidator | None = None,
    ) -> None:
        self.kind = _coerce_plugin_kind(kind)
        self._factories: dict[str, PluginFactory] = {}
        self._descriptors: dict[str, PluginDescriptor] = {}
        self._validator = validator
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def register(
        self,
        name: str,
        factory: PluginFactory,
        *,
        version: str = "1",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PluginDescriptor:
        if name in self._descriptors:
            raise AgentNetConfigurationError(
                f"{self.kind.value.title()} plugin {name!r} is already registered"
            )
        if not callable(factory):
            raise AgentNetConfigurationError("Plugin factory must be callable")

        descriptor = PluginDescriptor(
            name,
            self.kind,
            version=version,
            description=description,
            metadata=metadata,
        )
        self._descriptors[name] = descriptor
        self._factories[name] = factory
        return descriptor

    def register_descriptor(self, descriptor: PluginDescriptor) -> PluginDescriptor:
        if not isinstance(descriptor, PluginDescriptor):
            raise AgentNetConfigurationError(
                "Plugin registry descriptors must be PluginDescriptor instances"
            )
        if descriptor.kind != self.kind:
            raise AgentNetConfigurationError(
                f"Cannot register {descriptor.kind.value!r} descriptor "
                f"in {self.kind.value!r} registry"
            )
        if descriptor.name in self._descriptors:
            raise AgentNetConfigurationError(
                f"{self.kind.value.title()} plugin {descriptor.name!r} is already registered"
            )
        self._descriptors[descriptor.name] = descriptor
        return descriptor

    def descriptor(self, name: str) -> PluginDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(
                f"Unknown {self.kind.value} plugin {name!r}"
            ) from exc

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.descriptor(name)
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(
                f"{self.kind.value.title()} plugin {name!r} has no registered factory"
            ) from exc
        result = factory(*args, **kwargs)
        if self._validator is not None:
            self._validator(name, result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "plugins": [
                self._descriptors[name].to_dict() for name in sorted(self._descriptors)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        kind = payload.get("kind")
        if kind is None:
            raise AgentNetConfigurationError("Plugin registry payload requires kind")
        registry = cls(str(kind))
        for descriptor in payload.get("plugins", ()):
            registry.register_descriptor(PluginDescriptor.from_dict(dict(descriptor)))
        return registry


class ProviderPluginRegistry(PluginRegistry):
    """Registry for provider plugins that create LLM backends."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(
            PluginKind.PROVIDER,
            factories=factories,
            validator=_validate_provider_plugin,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.PROVIDER)
        return registry


class OptimizerPluginRegistry(PluginRegistry):
    """Registry for optimizer plugins."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(
            PluginKind.OPTIMIZER,
            factories=factories,
            validator=_validate_optimizer_plugin,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.OPTIMIZER)
        return registry


class EvaluatorPluginRegistry(PluginRegistry):
    """Registry for evaluator plugins that create objectives."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(
            PluginKind.EVALUATOR,
            factories=factories,
            validator=_validate_evaluator_plugin,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.EVALUATOR)
        return registry


class SchedulerPluginRegistry(PluginRegistry):
    """Registry for scheduler plugins."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(
            PluginKind.SCHEDULER,
            factories=factories,
            validator=_validate_scheduler_plugin,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.SCHEDULER)
        return registry


class TracerPluginRegistry(PluginRegistry):
    """Registry for tracer plugins."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(
            PluginKind.TRACER,
            factories=factories,
            validator=_validate_tracer_plugin,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.TRACER)
        return registry


class StoragePluginRegistry(PluginRegistry):
    """Registry for storage plugins."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(PluginKind.STORAGE, factories=factories)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.STORAGE)
        return registry


class MemoryPluginRegistry(PluginRegistry):
    """Registry for memory plugins."""

    def __init__(
        self,
        factories: Mapping[str, PluginFactory] | None = None,
    ) -> None:
        super().__init__(PluginKind.MEMORY, factories=factories)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        registry = cls()
        _register_descriptors_from_payload(registry, payload, PluginKind.MEMORY)
        return registry


class PluginManager:
    """Container for all AgentNet plugin registries."""

    def __init__(
        self,
        *,
        providers: ProviderPluginRegistry | None = None,
        optimizers: OptimizerPluginRegistry | None = None,
        evaluators: EvaluatorPluginRegistry | None = None,
        schedulers: SchedulerPluginRegistry | None = None,
        tracers: TracerPluginRegistry | None = None,
        storage: StoragePluginRegistry | None = None,
        memory: MemoryPluginRegistry | None = None,
    ) -> None:
        self.providers = providers or ProviderPluginRegistry()
        self.optimizers = optimizers or OptimizerPluginRegistry()
        self.evaluators = evaluators or EvaluatorPluginRegistry()
        self.schedulers = schedulers or SchedulerPluginRegistry()
        self.tracers = tracers or TracerPluginRegistry()
        self.storage = storage or StoragePluginRegistry()
        self.memory = memory or MemoryPluginRegistry()

    def registry(self, kind: PluginKind | str) -> PluginRegistry:
        plugin_kind = _coerce_plugin_kind(kind)
        return {
            PluginKind.PROVIDER: self.providers,
            PluginKind.OPTIMIZER: self.optimizers,
            PluginKind.EVALUATOR: self.evaluators,
            PluginKind.SCHEDULER: self.schedulers,
            PluginKind.TRACER: self.tracers,
            PluginKind.STORAGE: self.storage,
            PluginKind.MEMORY: self.memory,
        }[plugin_kind]

    def register(
        self,
        kind: PluginKind | str,
        name: str,
        factory: PluginFactory,
        *,
        version: str = "1",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PluginDescriptor:
        return self.registry(kind).register(
            name,
            factory,
            version=version,
            description=description,
            metadata=metadata,
        )

    def create(self, kind: PluginKind | str, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.registry(kind).create(name, *args, **kwargs)

    def register_provider(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.providers.register(name, factory, **kwargs)

    def register_optimizer(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.optimizers.register(name, factory, **kwargs)

    def register_evaluator(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.evaluators.register(name, factory, **kwargs)

    def register_scheduler(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.schedulers.register(name, factory, **kwargs)

    def register_tracer(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.tracers.register(name, factory, **kwargs)

    def register_storage(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.storage.register(name, factory, **kwargs)

    def register_memory(
        self,
        name: str,
        factory: PluginFactory,
        **kwargs: Any,
    ) -> PluginDescriptor:
        return self.memory.register(name, factory, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluators": self.evaluators.to_dict(),
            "memory": self.memory.to_dict(),
            "optimizers": self.optimizers.to_dict(),
            "providers": self.providers.to_dict(),
            "schedulers": self.schedulers.to_dict(),
            "storage": self.storage.to_dict(),
            "tracers": self.tracers.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            providers=ProviderPluginRegistry.from_dict(
                _registry_payload(payload, "providers", PluginKind.PROVIDER)
            ),
            optimizers=OptimizerPluginRegistry.from_dict(
                _registry_payload(payload, "optimizers", PluginKind.OPTIMIZER)
            ),
            evaluators=EvaluatorPluginRegistry.from_dict(
                _registry_payload(payload, "evaluators", PluginKind.EVALUATOR)
            ),
            schedulers=SchedulerPluginRegistry.from_dict(
                _registry_payload(payload, "schedulers", PluginKind.SCHEDULER)
            ),
            tracers=TracerPluginRegistry.from_dict(
                _registry_payload(payload, "tracers", PluginKind.TRACER)
            ),
            storage=StoragePluginRegistry.from_dict(
                _registry_payload(payload, "storage", PluginKind.STORAGE)
            ),
            memory=MemoryPluginRegistry.from_dict(
                _registry_payload(payload, "memory", PluginKind.MEMORY)
            ),
        )


def _coerce_plugin_kind(kind: PluginKind | str) -> PluginKind:
    try:
        return kind if isinstance(kind, PluginKind) else PluginKind(str(kind))
    except ValueError as exc:
        raise AgentNetConfigurationError(f"Unknown plugin kind {kind!r}") from exc


def _register_descriptors_from_payload(
    registry: PluginRegistry,
    payload: Mapping[str, Any],
    kind: PluginKind,
) -> None:
    payload_kind = payload.get("kind", kind.value)
    if _coerce_plugin_kind(str(payload_kind)) != kind:
        raise AgentNetConfigurationError(
            f"Plugin registry payload kind must be {kind.value!r}"
        )
    for descriptor in payload.get("plugins", ()):
        registry.register_descriptor(PluginDescriptor.from_dict(dict(descriptor)))


def _registry_payload(
    payload: Mapping[str, Any],
    key: str,
    kind: PluginKind,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {"kind": kind.value, "plugins": []}
    return dict(value)


def _validate_provider_plugin(name: str, plugin: Any) -> None:
    if not isinstance(plugin, LLMBackend):
        raise AgentNetConfigurationError(
            f"Provider plugin {name!r} did not return an LLMBackend"
        )


def _validate_optimizer_plugin(name: str, plugin: Any) -> None:
    if not callable(getattr(plugin, "optimize", None)) and not callable(
        getattr(plugin, "search", None)
    ):
        raise AgentNetConfigurationError(
            f"Optimizer plugin {name!r} did not return an optimizer"
        )


def _validate_evaluator_plugin(name: str, plugin: Any) -> None:
    if not isinstance(plugin, Objective):
        raise AgentNetConfigurationError(
            f"Evaluator plugin {name!r} did not return an Objective"
        )


def _validate_scheduler_plugin(name: str, plugin: Any) -> None:
    if not isinstance(plugin, Scheduler):
        raise AgentNetConfigurationError(
            f"Scheduler plugin {name!r} did not return a Scheduler"
        )


def _validate_tracer_plugin(name: str, plugin: Any) -> None:
    if not callable(getattr(plugin, "start_span", None)) or not callable(
        getattr(plugin, "end_span", None)
    ):
        raise AgentNetConfigurationError(
            f"Tracer plugin {name!r} did not return a tracer"
        )
