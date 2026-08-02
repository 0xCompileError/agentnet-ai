"""Communication interface definitions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, cast
from xml.etree import ElementTree

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.core.schema import validate_schema


@dataclass(frozen=True, slots=True)
class SemanticContractDescriptor:
    """Serializable semantic contract descriptor."""

    required_fields: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.copy(),
            "required_fields": list(self.required_fields),
        }

    @classmethod
    def from_dict(cls, descriptor: Mapping[str, Any]) -> SemanticContractDescriptor:
        return cls(
            required_fields=tuple(
                str(field) for field in descriptor.get("required_fields", ())
            ),
            metadata=dict(descriptor.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class RepresentationDescriptor:
    """Serializable representation descriptor."""

    identifier: str
    schema: str | None = None
    media_type: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "identifier": self.identifier,
            "media_type": self.media_type,
            "metadata": self.metadata.copy(),
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, descriptor: Mapping[str, Any]) -> RepresentationDescriptor:
        schema = descriptor.get("schema")
        return cls(
            identifier=str(descriptor["identifier"]),
            schema=None if schema is None else str(schema),
            media_type=descriptor.get("media_type"),
            description=descriptor.get("description"),
            metadata=dict(descriptor.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class InterfaceDescriptor:
    """Serializable, non-executable interface descriptor."""

    name: str | None = None
    schema: str | None = None
    description: str | None = None
    semantic_contract: SemanticContractDescriptor | None = None
    representations: tuple[RepresentationDescriptor, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "Interface"
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _coerce_interface_version(self.version))
        object.__setattr__(self, "representations", tuple(self.representations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "representations": [
                representation.to_dict() for representation in self.representations
            ],
            "schema": self.schema,
            "semantic_contract": (
                None
                if self.semantic_contract is None
                else self.semantic_contract.to_dict()
            ),
            "type": self.type,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, descriptor: Mapping[str, Any]) -> InterfaceDescriptor:
        semantic_contract = descriptor.get("semantic_contract")
        schema = descriptor.get("schema")
        return cls(
            name=descriptor.get("name"),
            schema=None if schema is None else str(schema),
            description=descriptor.get("description"),
            semantic_contract=(
                None
                if semantic_contract is None
                else SemanticContractDescriptor.from_dict(semantic_contract)
            ),
            representations=tuple(
                RepresentationDescriptor.from_dict(representation)
                for representation in descriptor.get("representations", ())
            ),
            metadata=dict(descriptor.get("metadata", {})),
            type=str(descriptor.get("type", "Interface")),
            version=str(descriptor.get("version", "1")),
        )


@dataclass(frozen=True, slots=True, init=False)
class SemanticContract:
    """Declarative semantic information required by an interface."""

    required_fields: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        required_fields: list[str] | tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        fields = tuple(str(field) for field in required_fields)
        if any(not field for field in fields):
            raise AgentNetConfigurationError(
                "SemanticContract required field names cannot be empty"
            )
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def validate(self, value: Any, *, label: str = "output") -> Any:
        if not self.required_fields:
            return value
        if not isinstance(value, Mapping):
            raise AgentNetValidationError(
                f"{label} must be a mapping to satisfy semantic contract"
            )
        for field_name in self.required_fields:
            if field_name not in value:
                raise AgentNetValidationError(
                    f"{label}.{field_name} is required by semantic contract"
                )
        return value

    def to_descriptor(self) -> SemanticContractDescriptor:
        return SemanticContractDescriptor(
            required_fields=self.required_fields,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_descriptor().to_dict()


@dataclass(frozen=True, slots=True, init=False)
class Representation:
    """Concrete format an interface value may use."""

    identifier: str
    schema: Any | None
    media_type: str | None
    description: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        identifier: str,
        *,
        schema: Any | None = None,
        media_type: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not identifier:
            raise AgentNetConfigurationError(
                "Representation identifier cannot be empty"
            )
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def validate(self, value: Any, *, label: str = "output") -> Any:
        return validate_schema(self.schema, value, label=label)

    def to_descriptor(self) -> RepresentationDescriptor:
        return RepresentationDescriptor(
            identifier=self.identifier,
            schema=_schema_repr(self.schema),
            media_type=self.media_type,
            description=self.description,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_descriptor().to_dict()


class RepresentationPluginRegistry:
    """Registry for explicitly provided custom representation factories."""

    def __init__(
        self,
        factories: Mapping[str, Callable[..., object]] | None = None,
    ) -> None:
        self._factories: dict[str, Callable[..., object]] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(self, name: str, factory: Callable[..., object]) -> None:
        if not name:
            raise AgentNetConfigurationError(
                "Representation plugin name cannot be empty"
            )
        if name in self._factories:
            raise AgentNetConfigurationError(
                f"Representation plugin {name!r} is already registered"
            )
        if not callable(factory):
            raise AgentNetConfigurationError(
                "Representation plugin factory must be callable"
            )
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Representation:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(
                f"Unknown representation plugin {name!r}"
            ) from exc

        representation = factory(**kwargs)
        if not isinstance(representation, Representation):
            raise AgentNetConfigurationError(
                f"Representation plugin {name!r} did not return a Representation"
            )
        return representation

    def to_dict(self) -> dict[str, list[str]]:
        return {"plugins": sorted(self._factories)}


class JSONSchemaRepresentation(Representation):
    """Representation validated with a small JSON Schema subset."""

    __slots__ = ("json_schema",)

    json_schema: dict[str, Any]

    def __init__(
        self,
        json_schema: Mapping[str, Any],
        *,
        identifier: str = "json_schema",
        media_type: str = "application/schema+json",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        schema_copy = copy.deepcopy(dict(json_schema))
        super().__init__(
            identifier,
            schema=schema_copy,
            media_type=media_type,
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "json_schema", schema_copy)

    def validate(self, value: Any, *, label: str = "output") -> Any:
        _validate_json_schema_value(self.json_schema, value, label)
        return value


class PydanticModelRepresentation(Representation):
    """Representation validated by a Pydantic-like model class."""

    __slots__ = ("model",)

    model: type[Any]

    def __init__(
        self,
        model: type[Any],
        *,
        identifier: str = "pydantic_model",
        media_type: str = "application/python-pydantic",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, type) or not _is_pydantic_like_model(model):
            raise AgentNetConfigurationError(
                "PydanticModelRepresentation requires a Pydantic-like model class"
            )
        super().__init__(
            identifier,
            schema=model,
            media_type=media_type,
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "model", model)

    def validate(self, value: Any, *, label: str = "output") -> Any:
        if isinstance(value, self.model):
            return value
        validator = getattr(self.model, "model_validate", None)
        if callable(validator):
            return _validate_with_pydantic_callable(validator, value, label)
        parser = getattr(self.model, "parse_obj", None)
        if callable(parser):
            return _validate_with_pydantic_callable(parser, value, label)
        raise AgentNetConfigurationError(
            "PydanticModelRepresentation model has no supported validator"
        )


class MarkdownRepresentation(Representation):
    """Markdown text representation."""

    __slots__ = ("require_heading",)

    require_heading: bool

    def __init__(
        self,
        *,
        identifier: str = "markdown",
        require_heading: bool = False,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            identifier,
            schema=str,
            media_type="text/markdown",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "require_heading", bool(require_heading))

    def validate(self, value: Any, *, label: str = "output") -> str:
        if not isinstance(value, str):
            raise AgentNetValidationError(f"{label} must be markdown text")
        if self.require_heading and not _contains_markdown_heading(value):
            raise AgentNetValidationError(f"{label} must include a markdown heading")
        return value


class PlainTextRepresentation(Representation):
    """Plain text representation."""

    __slots__ = ("require_non_empty",)

    require_non_empty: bool

    def __init__(
        self,
        *,
        identifier: str = "plain_text",
        require_non_empty: bool = False,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            identifier,
            schema=str,
            media_type="text/plain",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "require_non_empty", bool(require_non_empty))

    def validate(self, value: Any, *, label: str = "output") -> str:
        if not isinstance(value, str):
            raise AgentNetValidationError(f"{label} must be plain text")
        if self.require_non_empty and not value.strip():
            raise AgentNetValidationError(f"{label} must be non-empty plain text")
        return value


class BulletListRepresentation(Representation):
    """Bullet list representation as an ordered list of strings."""

    __slots__ = ("min_items",)

    min_items: int

    def __init__(
        self,
        *,
        identifier: str = "bullet_list",
        min_items: int = 0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if min_items < 0:
            raise AgentNetConfigurationError(
                "BulletListRepresentation min_items cannot be negative"
            )
        super().__init__(
            identifier,
            schema=list[str],
            media_type="application/vnd.agentnet.bullet-list",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "min_items", int(min_items))

    def validate(self, value: Any, *, label: str = "output") -> list[str]:
        if not isinstance(value, list):
            raise AgentNetValidationError(f"{label} must be a bullet list")
        if len(value) < self.min_items:
            raise AgentNetValidationError(
                f"{label} must contain at least {self.min_items} item(s)"
            )
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise AgentNetValidationError(f"{label}[{index}] must be text")
        return value


class XMLRepresentation(Representation):
    """XML text representation."""

    __slots__ = ("root_tag",)

    root_tag: str | None

    def __init__(
        self,
        *,
        identifier: str = "xml",
        root_tag: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            identifier,
            schema=str,
            media_type="application/xml",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "root_tag", root_tag)

    def validate(self, value: Any, *, label: str = "output") -> str:
        if not isinstance(value, str):
            raise AgentNetValidationError(f"{label} must be XML text")
        try:
            root = ElementTree.fromstring(value)
        except ElementTree.ParseError as exc:
            raise AgentNetValidationError(f"{label} must be well-formed XML") from exc
        if self.root_tag is not None and root.tag != self.root_tag:
            raise AgentNetValidationError(
                f"{label} root tag must be {self.root_tag!r}"
            )
        return value


class YAMLRepresentation(Representation):
    """YAML text representation."""

    __slots__ = ("require_mapping",)

    require_mapping: bool

    def __init__(
        self,
        *,
        identifier: str = "yaml",
        require_mapping: bool = False,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            identifier,
            schema=str,
            media_type="application/yaml",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "require_mapping", bool(require_mapping))

    def validate(self, value: Any, *, label: str = "output") -> str:
        if not isinstance(value, str):
            raise AgentNetValidationError(f"{label} must be YAML text")
        if self.require_mapping and not _contains_yaml_mapping(value):
            raise AgentNetValidationError(f"{label} must include a YAML mapping")
        return value


class KeyValueRepresentation(Representation):
    """Key-value mapping representation with string keys."""

    __slots__ = ("required_keys", "value_types")

    required_keys: tuple[str, ...]
    value_types: tuple[type[Any], ...] | None

    def __init__(
        self,
        *,
        identifier: str = "key_value",
        required_keys: Iterable[str] | str | None = None,
        value_types: type[Any] | Iterable[type[Any]] | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        key_tuple = _normalize_required_keys(required_keys)
        value_type_tuple = _normalize_value_types(value_types)
        super().__init__(
            identifier,
            schema=Mapping[str, Any],
            media_type="application/vnd.agentnet.key-value",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "required_keys", key_tuple)
        object.__setattr__(self, "value_types", value_type_tuple)

    def validate(self, value: Any, *, label: str = "output") -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise AgentNetValidationError(f"{label} must be a key-value mapping")
        for key, item in value.items():
            if not isinstance(key, str):
                raise AgentNetValidationError(f"{label} key must be text")
            if self.value_types is not None and not _matches_value_types(
                item,
                self.value_types,
            ):
                raise AgentNetValidationError(
                    f"{label}.{key} must match configured value_types"
                )
        for required_key in self.required_keys:
            if required_key not in value:
                raise AgentNetValidationError(f"{label}.{required_key} is required")
        return value


class EvidenceGraphRepresentation(Representation):
    """Evidence graph representation as nodes and directed evidence edges."""

    __slots__ = ("require_edges",)

    require_edges: bool

    def __init__(
        self,
        *,
        identifier: str = "evidence_graph",
        require_edges: bool = False,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            identifier,
            schema=Mapping[str, Any],
            media_type="application/vnd.agentnet.evidence-graph",
            description=description,
            metadata=metadata,
        )
        object.__setattr__(self, "require_edges", bool(require_edges))

    def validate(self, value: Any, *, label: str = "output") -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise AgentNetValidationError(f"{label} must be an evidence graph mapping")
        node_ids = _validate_evidence_graph_nodes(value, label)
        _validate_evidence_graph_edges(value, node_ids, label, self.require_edges)
        return value


@dataclass(frozen=True, slots=True, init=False)
class RepresentationTranslator:
    """Explicit in-process translator between two representations."""

    source: str
    target: str
    transform: Callable[[Any], Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        source: str,
        target: str,
        transform: object,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not source:
            raise AgentNetConfigurationError(
                "RepresentationTranslator source cannot be empty"
            )
        if not target:
            raise AgentNetConfigurationError(
                "RepresentationTranslator target cannot be empty"
            )
        if not callable(transform):
            raise AgentNetConfigurationError(
                "RepresentationTranslator transform must be callable"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "transform", cast(Callable[[Any], Any], transform))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def translate(self, value: Any) -> Any:
        return self.transform(value)


class RepresentationTranslatorRegistry:
    """In-memory registry of explicit representation translators."""

    def __init__(
        self,
        translators: Iterable[RepresentationTranslator] | None = None,
    ) -> None:
        self._translators: dict[tuple[str, str], RepresentationTranslator] = {}
        for translator in translators or ():
            self.register(translator)

    @property
    def translators(self) -> tuple[RepresentationTranslator, ...]:
        return tuple(self._translators.values())

    def register(self, translator: RepresentationTranslator) -> RepresentationTranslator:
        if not isinstance(translator, RepresentationTranslator):
            raise AgentNetConfigurationError(
                "Translator registry entries must be RepresentationTranslator instances"
            )
        self._translators[(translator.source, translator.target)] = translator
        return translator

    def get(self, source: str, target: str) -> RepresentationTranslator:
        try:
            return self._translators[(source, target)]
        except KeyError as exc:
            raise AgentNetValidationError(
                f"No translator registered for {source!r} -> {target!r}"
            ) from exc

    def translate(self, source: str, target: str, value: Any) -> Any:
        return self.get(source, target).translate(value)


@dataclass(frozen=True, slots=True, init=False)
class Interface:
    """Semantic output contract for agent communication."""

    schema: Any | None
    name: str | None
    description: str | None
    semantic_contract: SemanticContract | None
    representations: tuple[Representation, ...]
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        schema: Any | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        semantic_contract: SemanticContract | None = None,
        representations: Iterable[Representation] | None = None,
        version: str = "1",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if name == "":
            raise AgentNetConfigurationError("Interface name cannot be empty")
        representation_tuple = tuple(representations or ())
        _validate_representations(representation_tuple)
        version = _coerce_interface_version(version)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "semantic_contract", semantic_contract)
        object.__setattr__(self, "representations", representation_tuple)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def representation_identifiers(self) -> tuple[str, ...]:
        return tuple(representation.identifier for representation in self.representations)

    def supports_representation(self, identifier: str) -> bool:
        return identifier in self.representation_identifiers

    def get_representation(self, identifier: str) -> Representation:
        for representation in self.representations:
            if representation.identifier == identifier:
                return representation
        raise AgentNetValidationError(f"Unsupported representation {identifier!r}")

    def validate(
        self,
        value: Any,
        *,
        label: str = "output",
        representation: str | None = None,
    ) -> Any:
        validated = validate_schema(self.schema, value, label=label)
        if representation is not None:
            validated = self.get_representation(representation).validate(
                validated,
                label=label,
            )
        if self.semantic_contract is not None:
            self.semantic_contract.validate(validated, label=label)
        return validated

    def to_descriptor(self) -> InterfaceDescriptor:
        return InterfaceDescriptor(
            name=self.name,
            schema=_schema_repr(self.schema),
            description=self.description,
            semantic_contract=(
                None
                if self.semantic_contract is None
                else self.semantic_contract.to_descriptor()
            ),
            representations=tuple(
                representation.to_descriptor()
                for representation in self.representations
            ),
            metadata=self.metadata,
            version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_descriptor().to_dict()


def _validate_representations(representations: tuple[Representation, ...]) -> None:
    identifiers: set[str] = set()
    for representation in representations:
        if not isinstance(representation, Representation):
            raise AgentNetConfigurationError(
                "Interface representations must be Representation instances"
            )
        if representation.identifier in identifiers:
            raise AgentNetConfigurationError(
                f"Duplicate representation identifier {representation.identifier!r}"
            )
        identifiers.add(representation.identifier)


def _schema_repr(schema: Any | None) -> str | None:
    if schema is None:
        return None
    return repr(schema)


def _coerce_interface_version(version: str) -> str:
    version = str(version)
    if not version:
        raise AgentNetConfigurationError("Interface version cannot be empty")
    return version


def _validate_json_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    label: str,
) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        _validate_json_object_schema(schema, value, label)
        return
    if schema_type == "array":
        _validate_json_array_schema(schema, value, label)
        return
    if schema_type == "string":
        if not isinstance(value, str):
            raise AgentNetValidationError(f"{label} must be string")
        return
    if schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise AgentNetValidationError(f"{label} must be integer")
        return
    if schema_type == "number":
        if not isinstance(value, Real) or isinstance(value, bool):
            raise AgentNetValidationError(f"{label} must be number")
        return
    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise AgentNetValidationError(f"{label} must be boolean")
        return
    if schema_type == "null" and value is not None:
        raise AgentNetValidationError(f"{label} must be null")


def _validate_json_object_schema(
    schema: Mapping[str, Any],
    value: Any,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise AgentNetValidationError(f"{label} must be object")

    for field_name in schema.get("required", ()):
        field_name = str(field_name)
        if field_name not in value:
            raise AgentNetValidationError(f"{label}.{field_name} is required")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return
    for field_name, field_schema in properties.items():
        field_name = str(field_name)
        if field_name not in value or not isinstance(field_schema, Mapping):
            continue
        _validate_json_schema_value(
            field_schema,
            value[field_name],
            f"{label}.{field_name}",
        )


def _validate_json_array_schema(
    schema: Mapping[str, Any],
    value: Any,
    label: str,
) -> None:
    if not isinstance(value, list):
        raise AgentNetValidationError(f"{label} must be array")
    item_schema = schema.get("items")
    if not isinstance(item_schema, Mapping):
        return
    for index, item in enumerate(value):
        _validate_json_schema_value(item_schema, item, f"{label}[{index}]")


def _is_pydantic_like_model(model: type[Any]) -> bool:
    return callable(getattr(model, "model_validate", None)) or callable(
        getattr(model, "parse_obj", None)
    )


def _validate_with_pydantic_callable(
    validator: Callable[[Any], Any],
    value: Any,
    label: str,
) -> Any:
    try:
        return validator(value)
    except Exception as exc:
        raise AgentNetValidationError(f"{label} failed Pydantic validation") from exc


def _contains_markdown_heading(value: str) -> bool:
    return any(line.startswith("# ") or line.startswith("## ") for line in value.splitlines())


def _contains_yaml_mapping(value: str) -> bool:
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        if ":" in stripped and not stripped.startswith(":"):
            return True
    return False


def _normalize_required_keys(required_keys: Iterable[str] | str | None) -> tuple[str, ...]:
    if required_keys is None:
        return ()
    keys = (required_keys,) if isinstance(required_keys, str) else tuple(required_keys)
    normalized = tuple(str(key) for key in keys)
    if any(not key for key in normalized):
        raise AgentNetConfigurationError(
            "KeyValueRepresentation required key names cannot be empty"
        )
    if len(set(normalized)) != len(normalized):
        raise AgentNetConfigurationError(
            "KeyValueRepresentation required key names cannot contain duplicates"
        )
    return normalized


def _normalize_value_types(
    value_types: type[Any] | Iterable[type[Any]] | None,
) -> tuple[type[Any], ...] | None:
    if value_types is None:
        return None
    types = (value_types,) if isinstance(value_types, type) else tuple(value_types)
    if not types:
        raise AgentNetConfigurationError(
            "KeyValueRepresentation value_types cannot be empty"
        )
    for value_type in types:
        if not isinstance(value_type, type):
            raise AgentNetConfigurationError(
                "KeyValueRepresentation value_types must contain types"
            )
    return types


def _matches_value_types(value: Any, value_types: tuple[type[Any], ...]) -> bool:
    if isinstance(value, bool) and bool not in value_types and int in value_types:
        return False
    return isinstance(value, value_types)


def _validate_evidence_graph_nodes(
    value: Mapping[Any, Any],
    label: str,
) -> set[str]:
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        raise AgentNetValidationError(f"{label}.nodes must be a list")
    if not nodes:
        raise AgentNetValidationError(
            f"{label}.nodes must contain at least one node"
        )

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise AgentNetValidationError(
                f"{label}.nodes[{index}] must be a mapping"
            )
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise AgentNetValidationError(
                f"{label}.nodes[{index}].id must be non-empty text"
            )
        if node_id in node_ids:
            raise AgentNetValidationError(
                f"{label}.nodes[{index}].id has duplicate node id {node_id!r}"
            )
        node_ids.add(node_id)
    return node_ids


def _validate_evidence_graph_edges(
    value: Mapping[Any, Any],
    node_ids: set[str],
    label: str,
    require_edges: bool,
) -> None:
    edges = value.get("edges", [])
    if not isinstance(edges, list):
        raise AgentNetValidationError(f"{label}.edges must be a list")
    if require_edges and not edges:
        raise AgentNetValidationError(
            f"{label}.edges must contain at least one edge"
        )
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise AgentNetValidationError(
                f"{label}.edges[{index}] must be a mapping"
            )
        source = _evidence_graph_edge_endpoint(edge, "source", index, label)
        target = _evidence_graph_edge_endpoint(edge, "target", index, label)
        for endpoint_name, endpoint in (("source", source), ("target", target)):
            if endpoint not in node_ids:
                raise AgentNetValidationError(
                    f"{label}.edges[{index}].{endpoint_name} references unknown node"
                )


def _evidence_graph_edge_endpoint(
    edge: Mapping[Any, Any],
    endpoint: str,
    index: int,
    label: str,
) -> str:
    value = edge.get(endpoint)
    if not isinstance(value, str) or not value:
        raise AgentNetValidationError(
            f"{label}.edges[{index}].{endpoint} must be non-empty text"
        )
    return value


@dataclass(frozen=True, slots=True)
class RepresentationNegotiation:
    """Result of choosing a representation between two interfaces."""

    source: Interface
    target: Interface
    representation: Representation | None

    @property
    def identifier(self) -> str | None:
        if self.representation is None:
            return None
        return self.representation.identifier


def negotiate_representation(
    source: Interface,
    target: Interface,
    *,
    preferred: Iterable[str] | None = None,
) -> RepresentationNegotiation:
    """Select a representation supported by both interfaces."""

    source_identifiers = source.representation_identifiers
    target_identifiers = target.representation_identifiers
    if not source_identifiers and not target_identifiers:
        return RepresentationNegotiation(source, target, None)

    compatible = _compatible_representation_identifiers(source, target)
    if not compatible:
        raise AgentNetValidationError("No compatible representation between interfaces")

    preferred_identifiers = tuple(preferred or ())
    for identifier in preferred_identifiers:
        if identifier in compatible:
            return RepresentationNegotiation(
                source,
                target,
                _representation_for_identifier(source, target, identifier),
            )

    identifier = compatible[0]
    return RepresentationNegotiation(
        source,
        target,
        _representation_for_identifier(source, target, identifier),
    )


def _compatible_representation_identifiers(
    source: Interface,
    target: Interface,
) -> tuple[str, ...]:
    source_identifiers = source.representation_identifiers
    target_identifiers = target.representation_identifiers
    if source_identifiers and target_identifiers:
        target_set = set(target_identifiers)
        return tuple(identifier for identifier in source_identifiers if identifier in target_set)
    return source_identifiers or target_identifiers


def _representation_for_identifier(
    source: Interface,
    target: Interface,
    identifier: str,
) -> Representation:
    if source.supports_representation(identifier):
        return source.get_representation(identifier)
    return target.get_representation(identifier)


@dataclass(frozen=True, slots=True)
class InterfaceCompatibility:
    """Validated compatibility between a source and target interface."""

    source: Interface
    target: Interface
    negotiation: RepresentationNegotiation

    @property
    def compatible(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class IncompatibleInterfaceDetection:
    """Detected compatibility state for a source-target interface pair."""

    source: Interface
    target: Interface
    compatibility: InterfaceCompatibility | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def compatible(self) -> bool:
        return self.compatibility is not None

    @property
    def incompatible(self) -> bool:
        return self.compatibility is None

    @property
    def representation(self) -> str | None:
        if self.compatibility is None:
            return None
        return self.compatibility.negotiation.identifier


@dataclass(frozen=True, slots=True)
class SemanticEquivalenceValidation:
    """Validated semantic equivalence between two representation values."""

    source_value: Any
    target_value: Any
    source_representation: str | None = None
    target_representation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    equivalent: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class InformationPreservationValidation:
    """Validated preservation of required information across values."""

    source_value: Any
    target_value: Any
    required_fields: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    preserved: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class LossyTranslationDetection:
    """Detected information loss for a translation attempt."""

    source_value: Any
    target_value: Any
    source_representation: str | None = None
    target_representation: str | None = None
    required_fields: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "losses", tuple(self.losses))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def lossy(self) -> bool:
        return bool(self.losses)


def validate_semantic_equivalence(
    source_value: Any,
    target_value: Any,
    *,
    source_representation: str | None = None,
    target_representation: str | None = None,
    comparator: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SemanticEquivalenceValidation:
    """Validate that two representation values preserve the same semantics."""

    if comparator is None:
        equivalent = _normalized_semantic_value(source_value) == _normalized_semantic_value(
            target_value
        )
    else:
        if not callable(comparator):
            raise AgentNetConfigurationError(
                "Semantic equivalence comparator must be callable"
            )
        equivalent = bool(cast(Callable[[Any, Any], bool], comparator)(
            source_value,
            target_value,
        ))

    if not equivalent:
        raise AgentNetValidationError(
            "Representation values are not semantically equivalent"
        )

    return SemanticEquivalenceValidation(
        source_value=source_value,
        target_value=target_value,
        source_representation=source_representation,
        target_representation=target_representation,
        metadata=dict(metadata or {}),
    )


def validate_information_preservation(
    source_value: Any,
    target_value: Any,
    *,
    required_fields: Iterable[str] | str | None = None,
    comparator: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InformationPreservationValidation:
    """Validate that required information survives a representation change."""

    fields = _normalize_preservation_fields(required_fields)
    if not fields:
        validate_semantic_equivalence(
            source_value,
            target_value,
            comparator=comparator,
        )
        return InformationPreservationValidation(
            source_value=source_value,
            target_value=target_value,
            required_fields=fields,
            metadata=dict(metadata or {}),
        )

    if not isinstance(source_value, Mapping) or not isinstance(target_value, Mapping):
        raise AgentNetValidationError(
            "Information preservation with required fields requires mappings"
        )

    for field_name in fields:
        if field_name not in source_value:
            raise AgentNetValidationError(
                f"Information field {field_name!r} is missing from source"
            )
        if field_name not in target_value:
            raise AgentNetValidationError(
                f"Information field {field_name!r} is missing from target"
            )
        try:
            validate_semantic_equivalence(
                source_value[field_name],
                target_value[field_name],
                comparator=comparator,
            )
        except AgentNetValidationError as exc:
            raise AgentNetValidationError(
                f"Information field {field_name!r} was not preserved"
            ) from exc

    return InformationPreservationValidation(
        source_value=source_value,
        target_value=target_value,
        required_fields=fields,
        metadata=dict(metadata or {}),
    )


def detect_lossy_translation(
    source_value: Any,
    target_value: Any,
    *,
    source_representation: str | None = None,
    target_representation: str | None = None,
    required_fields: Iterable[str] | str | None = None,
    comparator: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LossyTranslationDetection:
    """Detect whether a translated value loses required information."""

    fields = _normalize_preservation_fields(required_fields)
    try:
        validate_information_preservation(
            source_value,
            target_value,
            required_fields=fields,
            comparator=comparator,
        )
    except AgentNetValidationError as exc:
        losses = (str(exc),)
    else:
        losses = ()

    return LossyTranslationDetection(
        source_value=source_value,
        target_value=target_value,
        source_representation=source_representation,
        target_representation=target_representation,
        required_fields=fields,
        losses=losses,
        metadata=dict(metadata or {}),
    )


def validate_interface_compatibility(
    source: Interface,
    target: Interface,
    *,
    preferred: Iterable[str] | None = None,
) -> InterfaceCompatibility:
    """Validate that source output can satisfy target input expectations."""

    _validate_semantic_compatibility(source, target)
    negotiation = negotiate_representation(source, target, preferred=preferred)
    return InterfaceCompatibility(source, target, negotiation)


def detect_incompatible_interfaces(
    source: Interface,
    target: Interface,
    *,
    preferred: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IncompatibleInterfaceDetection:
    """Detect interface incompatibility without raising on validation failure."""

    try:
        compatibility = validate_interface_compatibility(
            source,
            target,
            preferred=preferred,
        )
    except AgentNetValidationError as exc:
        return IncompatibleInterfaceDetection(
            source=source,
            target=target,
            reason=str(exc),
            metadata=dict(metadata or {}),
        )
    return IncompatibleInterfaceDetection(
        source=source,
        target=target,
        compatibility=compatibility,
        metadata=dict(metadata or {}),
    )


def _validate_semantic_compatibility(source: Interface, target: Interface) -> None:
    source_fields = _semantic_required_fields(source)
    target_fields = _semantic_required_fields(target)
    missing = target_fields - source_fields
    if missing:
        fields = ", ".join(sorted(missing))
        raise AgentNetValidationError(
            f"Source interface does not provide required semantic fields: {fields}"
        )


def _semantic_required_fields(interface: Interface) -> set[str]:
    if interface.semantic_contract is None:
        return set()
    return set(interface.semantic_contract.required_fields)


def _normalized_semantic_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return {
            key: _normalized_semantic_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalized_semantic_value(item) for item in value]
    return value


def _normalize_preservation_fields(
    required_fields: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if required_fields is None:
        return ()
    fields = (required_fields,) if isinstance(required_fields, str) else tuple(required_fields)
    normalized = tuple(str(field) for field in fields)
    if any(not field for field in normalized):
        raise AgentNetConfigurationError(
            "Information preservation required field names cannot be empty"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RepresentationSelection:
    """Selected representation for a source-target interface pair."""

    source: Interface
    target: Interface
    representation: Representation | None
    score: float


def select_representation(
    source: Interface,
    target: Interface,
    *,
    scorer: Callable[[Representation], float] | None = None,
) -> RepresentationSelection:
    """Automatically select the best compatible representation."""

    if scorer is None:
        negotiation = negotiate_representation(source, target)
        return RepresentationSelection(
            source,
            target,
            negotiation.representation,
            0.0,
        )

    compatible = _compatible_representation_identifiers(source, target)
    if not compatible:
        raise AgentNetValidationError("No compatible representation between interfaces")

    best_representation: Representation | None = None
    best_score: float | None = None
    for identifier in compatible:
        representation = _representation_for_identifier(source, target, identifier)
        score = float(scorer(representation))
        if best_score is None or score > best_score:
            best_representation = representation
            best_score = score

    if best_representation is None or best_score is None:
        raise AgentNetValidationError("No compatible representation between interfaces")
    return RepresentationSelection(source, target, best_representation, best_score)
