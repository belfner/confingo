"""Dataclass-backed marshal / unmarshal core for config schemas.

Dataclasses are the single source of truth: one declaration serves at once as the
schema, the type validator, and the default values. ``from_dict`` builds a
dataclass tree from a nested mapping, coercing each value toward its annotated
type; ``to_dict`` renders a built config back into plain, serializable data.

Runs on Python 3.11 and newer. Generics use ``TypeVar`` and ``Generic`` so the
module stays importable on 3.11, which reaches end of life after the PEP 695
syntax it would otherwise use.

Supported field types:
Leaf types are ``bool``, ``int``, ``float``, ``str``, ``Path``, ``datetime`` /
``date`` / ``time``, ``Enum`` subclasses, ``Literal[...]``, ``Any``, and ``None``.
Composite types are nested dataclasses, ``list`` / ``tuple`` / ``set`` /
``frozenset`` / ``Sequence`` of a supported type, ``dict[str, X]`` / ``Mapping``
with ``str`` keys, and unions of supported types. ``Enum`` members and ``Literal``
arguments carry primitive values (``str`` / ``int`` / ``bool``) so they round-trip
through plain data. An ``init=True`` field annotated with a type outside this set
is reported as an issue; an ``init=False`` field holds runtime state populated in
``__post_init__``, so its annotation is exempt from this boundary and may name any
resolvable type.

Field options:
``init`` is the master switch. An ``init=False`` field is excluded from loading,
export, equality, and the fingerprint, and is populated in ``__post_init__``
(checked for completeness after construction). On an ``init=True`` field
``compare=False`` drops the field from equality and the fingerprint, and
``hash=False`` drops it from the fingerprint alone; ``hash=True`` with
``compare=False`` is reported as a contradiction.

Resolution order:
Values are layered lowest to highest precedence:

1. dataclass field defaults,
2. the mapping passed to ``from_dict`` (typically parsed from a config file).

A field absent from the mapping falls back to its declared default, used as the
author wrote it; defaults are trusted rather than re-coerced. A dataclass-typed
field with no default builds implicitly from an empty mapping, recursively, so
its own required values are reported at their nested dotted paths. Every other
field with no default is required.

Validation:
``from_dict`` walks the whole dataclass tree before raising, so one call reports
every problem it found: unknown keys, missing required values, type mismatches,
and a ``ValueError`` or ``TypeError`` raised from ``__post_init__``. To report
several problems from a single node, give the dataclass an ``__validate__`` method
returning an iterable of message strings; each becomes its own entry in the report.

Identity:
``config_hash`` fingerprints the resolved config with a stable digest over its
canonical JSON form. The digest is stable across processes and independent of
mapping key order and set iteration order, which makes it usable for run naming,
deduplication, and confirming that a rerun used the same settings.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
import json
import math
import types
import typing
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import (
    MISSING,
    Field,
    dataclass,
    fields,
    is_dataclass,
)
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from confingo import _arrays


T = TypeVar("T")

_UNSET = object()
"""Sentinel returned by coercion helpers when a value failed to convert."""

_HINT_CACHE: dict[type[Any], dict[str, Any]] = {}

_SCHEMA_CACHE: dict[type[Any], tuple[ConfigIssue, ...]] = {}
"""Per-dataclass cache of schema-validation issues, keyed by the root type."""

_BARE_CONTAINERS: dict[Any, Any] = {
    tuple: tuple,
    list: list,
    set: set,
    frozenset: frozenset,
    dict: dict,
    Sequence: list,
    Mapping: dict,
}
"""Container annotations written without element types, mapped to the type they build.

Membership is by exact identity, so a ``NamedTuple`` subclass resolves through the
scalar path and keeps its own type.
"""

_SEQUENCE_BUILDERS: dict[Any, Any] = {tuple: tuple, set: set, frozenset: frozenset}
"""Sequence origins mapped to their builder; any other origin builds a ``list``."""

_CONTAINER_ORIGINS: frozenset[Any] = frozenset({list, tuple, set, frozenset, dict, Sequence, Mapping})
"""Parameterized generic origins the engine accepts; every other origin is rejected."""


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigIssue:
    """A single problem found while building a config.

    Args:
        path: Dotted path to the offending value, such as ``training.trainer.lr``.
          The root config itself carries an empty path, rendered as ``<root>``.
        message: Human-readable description of the problem.
    """

    path: str
    message: str

    def __str__(self) -> str:
        label = "<root>" if self.path == "" else self.path
        return f"{label}: {self.message}"


class ConfigError(ValueError):
    """Raised when a config fails to build, carrying every issue found.

    Args:
        issues: The problems collected during the build.
        context: Description of the config source, used in the summary line.

    Attributes:
        issues: The problems collected during the build, in discovery order.
        context: Description of the config source.
    """

    def __init__(self, issues: Iterable[ConfigIssue], *, context: str) -> None:
        self.issues = tuple(issues)
        self.context = context
        count = len(self.issues)
        noun = "issue" if count == 1 else "issues"
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"{context} has {count} {noun}:\n{detail}")

    @classmethod
    def single(cls, message: str, *, context: str, path: str = "") -> ConfigError:
        """Build an error carrying exactly one issue.

        Args:
            message: Human-readable description of the problem.
            context: Description of the config source, used in the summary line.
            path: Dotted path to the offending value, by default the root.

        Returns:
            An error whose ``issues`` holds the single problem.
        """
        return cls([ConfigIssue(path=path, message=message)], context=context)


class _IssueCollector:
    """Accumulates config issues so one build reports all of them at once."""

    def __init__(self) -> None:
        self.issues: list[ConfigIssue] = []

    def add(self, path: str, message: str) -> None:
        """Record one issue.

        Args:
            path: Dotted path to the offending value.
            message: Human-readable description of the problem.
        """
        self.issues.append(ConfigIssue(path=path, message=message))

    def clean(self) -> bool:
        """Report whether the build has stayed issue-free so far.

        Returns:
            True while no issue has been recorded.
        """
        return len(self.issues) == 0


def _reject(collector: _IssueCollector, path: str, message: str) -> Any:
    """Record one issue and return the coercion-failure sentinel.

    Bundles the two steps that end every failed coercion branch so a caller
    writes ``return _reject(collector, path, message)`` on one line.

    Args:
        collector: Destination for the issue.
        path: Dotted path to the offending value.
        message: Human-readable description of the problem.

    Returns:
        The ``_UNSET`` sentinel.
    """
    collector.add(path, message)
    return _UNSET


# ---------------------------------------------------------------------------
# Type-hint helpers
# ---------------------------------------------------------------------------


def _resolved_hints(config_cls: type[Any]) -> dict[str, Any]:
    """Resolve a dataclass's annotations to runtime type objects, with caching.

    Args:
        config_cls: The dataclass whose annotations to resolve.

    Returns:
        Mapping of field name to resolved type hint, with ``Annotated`` metadata
        stripped so decorated fields resolve to their base type.

    Raises:
        ConfigError: When an annotation names a type that is unreachable from the
          defining module's namespace. Declare config dataclasses at module level so
          every name they reference resolves there.
    """
    cached = _HINT_CACHE.get(config_cls)
    if cached is not None:
        return cached
    try:
        hints = get_type_hints(config_cls, include_extras=True)
    except NameError as exc:
        message = (
            f"cannot resolve the annotations of {config_cls.__name__}: {exc}. "
            f"Declare config dataclasses at module level so their annotations resolve "
            f"in the defining module's namespace."
        )
        raise ConfigError.single(message, context="config schema") from exc
    _HINT_CACHE[config_cls] = hints
    from confingo._equality import _install_canonical_eq  # noqa: PLC0415

    _install_canonical_eq(config_cls)
    return hints


def _strip_annotated(hint: Any) -> Any:
    """Unwrap ``Annotated`` layers, returning the underlying type hint.

    Args:
        hint: A resolved type hint, possibly wrapped in ``Annotated``.

    Returns:
        The underlying hint; hints without metadata return unchanged.
    """
    while get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return hint


def _is_dataclass_type(hint: Any) -> bool:
    """Report whether a type hint is a dataclass type.

    Args:
        hint: The resolved type hint to inspect.

    Returns:
        True when the hint is a dataclass type rather than an instance.
    """
    return isinstance(hint, type) and is_dataclass(hint)


@dataclass(frozen=True)
class _ClassifiedField:
    """One dataclass field labelled for every projection the engine reads.

    Attributes:
        definition: The underlying ``dataclasses.Field``.
        has_default: Whether the field carries a default or default_factory.
        loadable: Whether the field is built from config input (``field.init``).
        exportable: Whether ``to_dict`` emits the field (``field.init``).
        compared: Whether equality includes the field (``init and compare``).
        hashed: Whether ``config_hash`` includes the field
          (``init and compare and effective_hash``).
        conflicts: Contradictory-flag messages for the field, empty when valid.
    """

    definition: Field[Any]
    has_default: bool
    loadable: bool
    exportable: bool
    compared: bool
    hashed: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class _DataclassFields:
    """A dataclass's fields grouped by projection, computed once and cached.

    Attributes:
        declared: Every field in declaration order.
        loadable: Fields built from config input (``init=True``).
        non_init: Fields excluded from construction (``init=False``), populated
          in ``__post_init__`` and checked for completeness after construction.
        exportable: Fields ``to_dict`` emits.
        compared: Fields equality includes.
        hashed: Fields ``config_hash`` includes.
        by_name: Every declared field keyed by name.
    """

    declared: tuple[_ClassifiedField, ...]
    loadable: tuple[_ClassifiedField, ...]
    non_init: tuple[_ClassifiedField, ...]
    exportable: tuple[_ClassifiedField, ...]
    compared: tuple[_ClassifiedField, ...]
    hashed: tuple[_ClassifiedField, ...]
    by_name: Mapping[str, _ClassifiedField]


_FIELD_CACHE: dict[type[Any], _DataclassFields] = {}
"""Per-dataclass cache of field classifications, keyed by class identity."""


def _classify_dataclass_fields(config_cls: type[Any]) -> _DataclassFields:
    """Classify a dataclass's fields for every engine projection, with caching.

    ``init`` is the master switch: an ``init=False`` field is excluded from
    loading, export, equality, and the fingerprint, so its ``compare`` / ``hash``
    flags are inert. On an ``init=True`` field ``compare`` scopes equality (and
    therefore the fingerprint) and ``hash`` scopes the fingerprint, with the one
    contradiction ``hash=True, compare=False`` recorded in ``conflicts``.

    Args:
        config_cls: The dataclass to classify.

    Returns:
        The grouped classification, cached by class identity.
    """
    cached = _FIELD_CACHE.get(config_cls)
    if cached is not None:
        return cached
    declared: list[_ClassifiedField] = []
    for field in fields(config_cls):
        effective_hash = field.compare if field.hash is None else field.hash
        conflicts: list[str] = []
        if field.init and field.hash is True and field.compare is False:
            conflicts.append(
                "field(hash=True, compare=False) is contradictory: config_hash fields must participate in equality"
            )
        declared.append(
            _ClassifiedField(
                definition=field,
                has_default=field.default is not MISSING or field.default_factory is not MISSING,
                loadable=field.init,
                exportable=field.init,
                compared=field.init and field.compare,
                hashed=field.init and field.compare and effective_hash,
                conflicts=tuple(conflicts),
            )
        )
    result = _DataclassFields(
        declared=tuple(declared),
        loadable=tuple(item for item in declared if item.loadable),
        non_init=tuple(item for item in declared if not item.definition.init),
        exportable=tuple(item for item in declared if item.exportable),
        compared=tuple(item for item in declared if item.compared),
        hashed=tuple(item for item in declared if item.hashed),
        by_name={item.definition.name: item for item in declared},
    )
    _FIELD_CACHE[config_cls] = result
    return result


def _projected_fields(config_cls: type[Any], projection: _PlainProjection) -> tuple[_ClassifiedField, ...]:
    """Select the classified fields a serialization projection emits.

    Args:
        config_cls: The dataclass whose fields to select.
        projection: The serialization projection.

    Returns:
        The classified fields for the projection, in declaration order.
    """
    classification = _classify_dataclass_fields(config_cls)
    if projection is _PlainProjection.EXPORT:
        return classification.exportable
    if projection is _PlainProjection.COMPARE:
        return classification.compared
    return classification.hashed


def _validate_schema(config_cls: type[Any]) -> tuple[ConfigIssue, ...]:
    """Validate a dataclass's field annotations against the supported type set.

    This inspects the schema itself, independent of any config data, so an
    unsupported annotation is reported even when the field is omitted and falls
    back to its default. Field default values are left untouched.

    Args:
        config_cls: The root dataclass to validate.

    Returns:
        The schema issues found, empty when the schema is fully supported.
    """
    cached = _SCHEMA_CACHE.get(config_cls)
    if cached is not None:
        return cached
    issues: list[ConfigIssue] = []
    _validate_dataclass_schema(config_cls, "", issues, set())
    result = tuple(issues)
    _SCHEMA_CACHE[config_cls] = result
    return result


def _validate_dataclass_schema(
    config_cls: type[Any], path: str, issues: list[ConfigIssue], seen: set[type[Any]]
) -> None:
    """Collect schema issues for one dataclass, recursing into nested dataclasses.

    Args:
        config_cls: The dataclass to inspect.
        path: Dotted schema path of this node, empty at the root.
        issues: Destination for any schema issues found.
        seen: Dataclasses already visited on this path, to break reference cycles.
    """
    if config_cls in seen:
        return
    seen = seen | {config_cls}
    hints = _resolved_hints(config_cls)
    for classified in _classify_dataclass_fields(config_cls).declared:
        field = classified.definition
        field_path = _join(path, field.name)
        issues.extend(ConfigIssue(field_path, message) for message in classified.conflicts)
        if not field.init:
            # init=False fields are runtime state exempt from the supported-type
            # boundary; their annotation need only resolve, checked by
            # _resolved_hints above.
            continue
        _validate_hint_schema(hints[field.name], field_path, issues, seen)


def _validate_hint_schema(hint: Any, path: str, issues: list[ConfigIssue], seen: set[type[Any]]) -> None:
    """Collect schema issues for one resolved type hint.

    Args:
        hint: The resolved type hint to inspect.
        path: Dotted schema path of the field carrying this hint.
        issues: Destination for any schema issues found.
        seen: Dataclasses already visited on this path, to break reference cycles.
    """
    array_match = _arrays.inspect_annotation(hint)
    if array_match.matched:
        if array_match.error is not None:
            issues.append(ConfigIssue(path, array_match.error))
        return
    hint = _strip_annotated(hint)
    if hint is Any or hint is type(None):
        return

    origin = get_origin(hint)
    args = get_args(hint)

    if origin is Literal:
        # Exact type, not isinstance: an Enum member can subclass str/int yet fails
        # the exact-type Literal match, so it is not a supported primitive option.
        issues.extend(
            ConfigIssue(path, f"Literal values must be primitive (str, int, bool); got {option!r}")
            for option in args
            if option is not None and type(option) not in (bool, int, str)
        )
        return

    if origin is typing.Union or origin is types.UnionType:
        for member in args:
            _validate_hint_schema(member, path, issues, seen)
        return

    if origin is not None:
        if origin not in _CONTAINER_ORIGINS:
            issues.append(ConfigIssue(path, f"unsupported field type {_hint_name(hint)}"))
            return
        if origin in (dict, Mapping):
            key_hint = args[0] if len(args) == 2 else str
            value_hint = args[1] if len(args) == 2 else Any
            if key_hint is not str:
                message = f"unsupported dict key type {_hint_name(key_hint)}; only str keys are supported"
                issues.append(ConfigIssue(path, message))
            _validate_hint_schema(value_hint, path, issues, seen)
            return
        for element_hint in args:
            if element_hint is not Ellipsis:
                _validate_hint_schema(element_hint, path, issues, seen)
        return

    if _is_dataclass_type(hint):
        _validate_dataclass_schema(hint, path, issues, seen)
        return

    if isinstance(hint, type):
        if hint in _BARE_CONTAINERS:
            return
        if issubclass(hint, Enum):
            for member in hint:
                if not isinstance(member.value, (bool, int, str)):
                    issues.append(
                        ConfigIssue(
                            path, f"enum {hint.__name__} must carry primitive values; {member.name} is {member.value!r}"
                        )
                    )
                    break
            return
        if hint in (bool, int, float, str) or issubclass(hint, (Path, dt.date, dt.time)):
            return

    issues.append(ConfigIssue(path, f"unsupported field type {_hint_name(hint)}"))


def _hint_name(hint: Any) -> str:
    """Render a type hint as a short readable name for error messages.

    Args:
        hint: The resolved type hint to describe.

    Returns:
        A display name such as ``int``, ``str | None``, or ``list[Path]``.
    """
    hint = _strip_annotated(hint)
    if hint is type(None):
        return "None"
    if hint is Ellipsis:
        return "..."
    origin = get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(_hint_name(arg) for arg in get_args(hint))
    if origin is Literal:
        return " | ".join(repr(arg) for arg in get_args(hint))
    args = get_args(hint)
    if origin is not None and len(args) > 0:
        base = getattr(origin, "__name__", str(origin))
        return f"{base}[{', '.join(_hint_name(arg) for arg in args)}]"
    return str(getattr(hint, "__name__", hint))


def _typename(value: Any) -> str:
    """Name the runtime type of a value for error messages.

    Args:
        value: The value to describe.

    Returns:
        The type name, with ``None`` reported as ``None``.
    """
    return "None" if value is None else type(value).__name__


def _join(path: str, key: str) -> str:
    """Append a key to a dotted config path.

    Args:
        path: The parent dotted path, empty at the root.
        key: The child key or index to append.

    Returns:
        The combined dotted path.
    """
    return key if path == "" else f"{path}.{key}"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def from_dict(config_cls: type[T], data: Mapping[str, Any], *, context: str = "config") -> T:
    """Build a dataclass tree from a nested mapping, reporting every problem at once.

    Walks ``config_cls`` by introspection, recursing into nested dataclasses and
    into dataclasses held in lists, tuples, sets, and dict values. Scalars are
    coerced toward the annotated type: sequences become the annotated container,
    strings and values resolve to ``Enum`` members, ``str`` becomes ``Path``,
    integral floats become ``int`` so forms like ``1e6`` land on ``int`` fields,
    ISO 8601 strings resolve to ``datetime`` / ``date`` / ``time``, and ``Literal``
    membership is checked. Dict fields carry ``str`` keys. A field whose annotation
    names a type outside the supported set is reported as an issue.

    An absent dataclass section builds implicitly from an empty mapping
    (recursively), so the section's own required leaves are reported at their
    nested dotted paths. Every other ``init=True`` field without a default is
    required when absent, container fields included, which keeps a forgotten
    container distinct from an intentionally empty one authored as
    ``field(default_factory=list)``. Explicit defaults and factories take
    precedence and are used as authored.

    An ``init=False`` field is runtime state: it is not built from the mapping
    (supplying its key is reported as not configurable), and it is populated by
    its default or in ``__post_init__``. After each node is constructed, every
    ``init=False`` field is checked for population, so one left unset by
    ``__post_init__`` is reported rather than surfacing later as an
    ``AttributeError``.

    Args:
        config_cls: The root dataclass to build.
        data: Nested mapping of config values, typically parsed from a config file.
        context: Description of the config source used in the error summary, by
          default ``"config"``.

    Returns:
        The constructed config object.

    Raises:
        ConfigError: When any key is unknown or not configurable, any required
          value is missing, any value fails to coerce, an ``init=False`` field is
          left unset by ``__post_init__``, or any node's ``__post_init__`` or
          ``__validate__`` rejects it. The exception lists every issue found in
          the whole tree.
    """
    schema_issues = _validate_schema(config_cls)
    if len(schema_issues) > 0:
        raise ConfigError(schema_issues, context=context)
    collector = _IssueCollector()
    instance = _build(config_cls, data, "", collector)
    if not collector.clean():
        raise ConfigError(collector.issues, context=context)
    return typing.cast("T", instance)


def _build(
    config_cls: type[Any],
    data: Any,
    path: str,
    collector: _IssueCollector,
    implicit_chain: tuple[type[Any], ...] = (),
) -> Any:
    """Construct one dataclass node, recording issues rather than raising.

    Args:
        config_cls: The dataclass to construct.
        data: The mapping of values for this node.
        path: Dotted path of this node, empty at the root.
        collector: Destination for any issues found.
        implicit_chain: Dataclass types currently being built implicitly on this
          branch, used to terminate self-referential schemas, by default ``()``.

    Returns:
        The constructed instance, or the ``_UNSET`` sentinel when this node failed
        to build.
    """
    if not isinstance(data, Mapping):
        return _reject(collector, path, f"expected a mapping for {config_cls.__name__}, got {_typename(data)}")

    hints = _resolved_hints(config_cls)
    classification = _classify_dataclass_fields(config_cls)
    loadable_names = {item.definition.name for item in classification.loadable}
    for key in data:
        if key in classification.by_name and key not in loadable_names:
            collector.add(_join(path, str(key)), "field is not configurable (init=False)")
        elif key not in loadable_names:
            collector.add(_join(path, str(key)), f"unknown key (known keys: {', '.join(sorted(loadable_names))})")

    kwargs: dict[str, Any] = {}
    node_failed = False
    for classified in classification.loadable:
        field = classified.definition
        field_path = _join(path, field.name)
        if field.name not in data:
            if not classified.has_default:
                hint = _strip_annotated(hints[field.name])
                if _is_dataclass_type(hint):
                    # An absent dataclass section builds implicitly from an empty
                    # mapping, so its own required leaves surface at their nested
                    # paths. The chain terminates self-referential schemas.
                    if hint in implicit_chain:
                        collector.add(field_path, "missing required value")
                        node_failed = True
                        continue
                    built = _build(hint, {}, field_path, collector, (*implicit_chain, hint))
                    if built is _UNSET:
                        node_failed = True
                        continue
                    kwargs[field.name] = built
                    continue
                collector.add(field_path, "missing required value")
                node_failed = True
            continue
        coerced = _coerce(data[field.name], hints[field.name], field_path, collector)
        if coerced is _UNSET:
            node_failed = True
            continue
        kwargs[field.name] = coerced

    # Walk every field before bailing so one build surfaces all of a node's
    # problems at once; kwargs would be incomplete once any field failed.
    if node_failed:
        return _UNSET

    try:
        instance = config_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        return _reject(collector, path, str(exc))

    # Every init=False field must be populated by its default or by
    # __post_init__ (which runs inside the constructor above) before __validate__
    # or user code reads it. object.__getattribute__ probes the real attribute,
    # bypassing any __getattr__ fallback, for both ordinary and slots classes.
    node_incomplete = False
    for classified in classification.non_init:
        name = classified.definition.name
        try:
            object.__getattribute__(instance, name)
        except AttributeError:
            collector.add(_join(path, name), "init=False field was not set during __post_init__")
            node_incomplete = True
    if node_incomplete:
        return _UNSET

    validate = getattr(instance, "__validate__", None)
    if callable(validate):
        for message in typing.cast("Iterable[Any]", validate()):
            collector.add(path, str(message))
    return instance


def _coerce(value: Any, hint: Any, path: str, collector: _IssueCollector) -> Any:
    """Convert one value toward its annotated type, recording issues on failure.

    Args:
        value: The raw value from the config mapping.
        hint: The resolved type hint the value must satisfy.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The coerced value, or the ``_UNSET`` sentinel when coercion failed.
    """
    if hint is Any:
        return _coerce_any(value, path, collector)

    array_match = _arrays.inspect_annotation(hint)
    if array_match.matched:
        if array_match.spec is None:
            return _reject(collector, path, typing.cast("str", array_match.error))
        result = _arrays.coerce_array(value, array_match.spec, path, collector.add)
        return _UNSET if result is _arrays.FAILED else result
    hint = _strip_annotated(hint)

    if hint is Any:
        return _coerce_any(value, path, collector)
    if hint is type(None):
        if value is None:
            return None
        return _reject(collector, path, f"expected None, got {_typename(value)}")

    origin = get_origin(hint)
    args = get_args(hint)

    if origin is Literal:
        # Exact-type match keeps bool True distinct from int 1, which compare equal.
        if any(value == option and type(value) is type(option) for option in args):
            return value
        return _reject(collector, path, f"expected one of {_hint_name(hint)}, got {value!r}")

    if origin is typing.Union or origin is types.UnionType:
        return _coerce_union(value, hint, args, path, collector)

    if origin is not None:
        if origin in _CONTAINER_ORIGINS:
            return _coerce_container(value, hint, origin, args, path, collector)
        return _reject(collector, path, f"unsupported field type {_hint_name(hint)}")

    if _is_dataclass_type(hint):
        return _build(hint, value, path, collector)

    if isinstance(hint, type):
        bare_origin = _BARE_CONTAINERS.get(hint)
        if bare_origin is not None:
            return _coerce_container(value, hint, bare_origin, (), path, collector)

    return _coerce_scalar(value, hint, path, collector)


def _coerce_any(value: Any, path: str, collector: _IssueCollector) -> Any:
    """Accept a value under an ``Any`` field, rejecting only non-finite floats.

    ``Any`` passes plain data through unchanged, but a non-finite float has no JSON
    form, so it is rejected wherever it appears in the accepted value, including
    inside nested mappings and sequences.

    Args:
        value: The raw value from the config mapping.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The value unchanged, or the ``_UNSET`` sentinel when it holds a non-finite
        float.
    """
    array_result = _arrays.validate_array_value(value, path, collector.add)
    if array_result is not _arrays.NOT_ARRAY:
        return _UNSET if array_result is _arrays.FAILED else array_result
    is_numpy_scalar, normalized = _arrays.normalize_numpy_scalar(value)
    if is_numpy_scalar:
        # Supported numpy scalars stay in memory as written and serialize as
        # Python scalars; only finiteness is enforced here.
        if isinstance(normalized, float) and not math.isfinite(normalized):
            return _reject(collector, path, f"expected a finite float, got {normalized!r}")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return _reject(collector, path, f"expected a finite float, got {value!r}")
    if isinstance(value, Mapping):
        failed = False
        for key, item in value.items():
            item_path = _join(path, str(key))
            if _arrays.is_array_value(key):
                # Arrays serialize as lists, which have no mapping-key form.
                collector.add(item_path, f"cannot use {_typename(key)} as a mapping key")
                failed = True
            elif _coerce_any(key, item_path, collector) is _UNSET:
                failed = True
            if _coerce_any(item, item_path, collector) is _UNSET:
                failed = True
        return _UNSET if failed else value
    if isinstance(value, (list, tuple, set, frozenset)):
        failed = False
        for index, item in enumerate(value):
            if _coerce_any(item, _join(path, str(index)), collector) is _UNSET:
                failed = True
        return _UNSET if failed else value
    return value


def _coerce_union(value: Any, hint: Any, args: tuple[Any, ...], path: str, collector: _IssueCollector) -> Any:
    """Coerce a value against a union, accepting the first member that fits cleanly.

    Args:
        value: The raw value from the config mapping.
        hint: The union type hint, used for the error message.
        args: The union's member types.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The coerced value, or the ``_UNSET`` sentinel when no member matched.
    """
    if value is None and type(None) in args:
        return None
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1:
        # A single-type optional (``X | None``) has one real branch, so coerce
        # directly against it in a single pass: its own nested issues surface, and
        # its ``__post_init__`` / ``__validate__`` run exactly once.
        return _coerce(value, non_none[0], path, collector)
    for candidate in non_none:
        # Probe each member with a throwaway collector so member-level failures
        # stay silent; the first clean conversion wins. Member order is precedence.
        trial = _IssueCollector()
        result = _coerce(value, candidate, path, trial)
        if result is not _UNSET and trial.clean():
            return result
    return _reject(collector, path, f"expected {_hint_name(hint)}, got {_typename(value)}")


def _coerce_items(items: list[Any], element_hints: list[Any], path: str, collector: _IssueCollector) -> Any:
    """Coerce positional items against per-index hints, one dotted path each.

    Args:
        items: The raw elements to coerce.
        element_hints: The type hint for each element, aligned by index with
          ``items``.
        path: Dotted path of the container holding these items.
        collector: Destination for any issues found.

    Returns:
        The list of coerced items, or the ``_UNSET`` sentinel when any element
        failed. Every element is visited so one pass reports all of them.
    """
    coerced: list[Any] = []
    failed = False
    for index, (item, element_hint) in enumerate(zip(items, element_hints, strict=True)):
        result = _coerce(item, element_hint, _join(path, str(index)), collector)
        if result is _UNSET:
            failed = True
            continue
        coerced.append(result)
    return _UNSET if failed else coerced


def _coerce_container(
    value: Any,
    hint: Any,
    origin: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
) -> Any:
    """Coerce a value into an annotated container, recursing into its elements.

    Args:
        value: The raw value from the config mapping.
        hint: The container type hint, used for the error message.
        origin: The container's unsubscripted origin type.
        args: The container's element type arguments.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The coerced container, or the ``_UNSET`` sentinel when coercion failed.
    """
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            return _reject(collector, path, f"expected a mapping for {_hint_name(hint)}, got {_typename(value)}")
        # Config files carry string keys, so bare mappings default to str keys and
        # only str-keyed dicts are supported.
        key_hint = args[0] if len(args) == 2 else str
        value_hint = args[1] if len(args) == 2 else Any
        if key_hint is not str:
            return _reject(
                collector, path, f"unsupported dict key type {_hint_name(key_hint)}; only str keys are supported"
            )
        result: dict[Any, Any] = {}
        failed = False
        for raw_key, raw_value in value.items():
            item_path = _join(path, str(raw_key))
            coerced_key = _coerce(raw_key, key_hint, item_path, collector)
            coerced_value = _coerce(raw_value, value_hint, item_path, collector)
            if coerced_key is _UNSET or coerced_value is _UNSET:
                failed = True
                continue
            result[coerced_key] = coerced_value
        return _UNSET if failed else result

    # str and bytes are themselves sequences, so route them to the scalar path
    # as leaves and accept only true element containers here.
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, set, frozenset)):
        return _reject(collector, path, f"expected a sequence for {_hint_name(hint)}, got {_typename(value)}")

    items = list(value)

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[X, ...]: one element type applied to every item.
            element_hints: list[Any] = [args[0]] * len(items)
        elif len(args) == 0:
            if get_origin(hint) is tuple:
                # tuple[()] is the subscripted empty-tuple form, so enforce arity 0.
                if len(items) != 0:
                    return _reject(collector, path, f"expected 0 items for {_hint_name(hint)}, got {len(items)}")
                element_hints = []
            else:
                # Bare tuple: each item passes through under Any.
                element_hints = [Any] * len(items)
        else:
            # Fixed-length tuple: each position has its own type, so arity must match.
            if len(items) != len(args):
                return _reject(collector, path, f"expected {len(args)} items for {_hint_name(hint)}, got {len(items)}")
            element_hints = list(args)
    else:
        element_hint = args[0] if len(args) >= 1 else Any
        element_hints = [element_hint] * len(items)

    coerced_items = _coerce_items(items, element_hints, path, collector)
    if coerced_items is _UNSET:
        return _UNSET
    builder = _SEQUENCE_BUILDERS.get(origin, list)
    try:
        return builder(coerced_items)
    except TypeError as exc:
        # A set/frozenset of unhashable elements fails to build; report it as an
        # issue rather than letting the raw TypeError escape the collector.
        return _reject(collector, path, f"cannot build {_hint_name(hint)}: {exc}")


def _coerce_scalar(value: Any, hint: Any, path: str, collector: _IssueCollector) -> Any:
    """Coerce a value toward a plain (unparameterized) annotated type.

    Args:
        value: The raw value from the config mapping.
        hint: The scalar type hint.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The coerced value, or the ``_UNSET`` sentinel when coercion failed.
    """
    is_numpy_scalar, normalized = _arrays.normalize_numpy_scalar(value)
    if is_numpy_scalar:
        # A supported numpy scalar feeds the ordinary rules as its exact Python
        # equivalent, so np.float32 lands on float fields and np.int64 on int.
        value = normalized

    if not isinstance(hint, type):
        return _reject(collector, path, f"unsupported field type {_hint_name(hint)}")

    if issubclass(hint, Enum):
        try:
            return hint(value)
        except ValueError:
            pass
        if isinstance(value, str) and value in hint.__members__:
            return hint[value]
        options = ", ".join(repr(member.value) for member in hint)
        return _reject(collector, path, f"expected one of {options} for enum {hint.__name__}, got {value!r}")

    if hint is bool:
        if isinstance(value, bool):
            return value
        return _reject(collector, path, f"expected bool, got {_typename(value)}")

    if hint is int:
        # bool is an int subclass, so reject it explicitly to keep True off int fields.
        if isinstance(value, bool):
            return _reject(collector, path, f"expected int, got {_typename(value)}")
        if isinstance(value, int):
            return value
        # Forms like 1e6 parse as float; accept whole-number floats onto int fields.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return _reject(collector, path, f"expected int, got {_typename(value)}")

    if hint is float:
        if isinstance(value, bool):
            return _reject(collector, path, f"expected float, got {_typename(value)}")
        if isinstance(value, (int, float)):
            try:
                result = float(value)
            except OverflowError:
                return _reject(collector, path, f"value is too large to represent as a float: {value!r}")
            if not math.isfinite(result):
                return _reject(collector, path, f"expected a finite float, got {value!r}")
            return result
        return _reject(collector, path, f"expected float, got {_typename(value)}")

    if hint is str:
        if isinstance(value, str):
            return value
        return _reject(collector, path, f"expected str, got {_typename(value)}")

    if issubclass(hint, Path):
        if isinstance(value, (str, Path)):
            return Path(value)
        return _reject(collector, path, f"expected a path string, got {_typename(value)}")

    if issubclass(hint, dt.datetime):
        if isinstance(value, dt.datetime):
            return value
        if isinstance(value, str):
            try:
                return dt.datetime.fromisoformat(value)
            except ValueError:
                return _reject(collector, path, f"expected an ISO 8601 datetime string, got {value!r}")
        return _reject(collector, path, f"expected an ISO 8601 datetime string, got {_typename(value)}")

    if issubclass(hint, dt.date):
        # datetime is a subclass of date and is handled above, so a datetime on a
        # plain date field is a type mismatch rather than a silent truncation.
        if isinstance(value, dt.datetime):
            return _reject(collector, path, f"expected a date, got {_typename(value)}")
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value)
            except ValueError:
                return _reject(collector, path, f"expected an ISO 8601 date string, got {value!r}")
        return _reject(collector, path, f"expected an ISO 8601 date string, got {_typename(value)}")

    if issubclass(hint, dt.time):
        if isinstance(value, dt.time):
            return value
        if isinstance(value, str):
            try:
                return dt.time.fromisoformat(value)
            except ValueError:
                return _reject(collector, path, f"expected an ISO 8601 time string, got {value!r}")
        return _reject(collector, path, f"expected an ISO 8601 time string, got {_typename(value)}")

    return _reject(collector, path, f"unsupported field type {hint.__name__}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class _PlainProjection(enum.Enum):
    """Which dataclass fields a plain-data walk emits.

    Attributes:
        EXPORT: Every exported field (``init=True``); the ``to_dict`` view.
        COMPARE: Every compared field (``init=True and compare``); equality's
          serialized fallback.
        HASH: Every hashed field (``init=True and compare and effective_hash``);
          the ``config_hash`` fingerprint view.
    """

    EXPORT = enum.auto()
    COMPARE = enum.auto()
    HASH = enum.auto()


def _project_plain(value: Any, projection: _PlainProjection) -> Any:
    """Render a config value to plain data under one field projection.

    Args:
        value: The config object or nested value to convert.
        projection: The field projection selecting which dataclass fields emit.

    Returns:
        The converted plain-data structure.

    Raises:
        ConfigError: When a value's type falls outside the supported set and has
          no plain-data form, or holds a non-finite float; the exception lists
          every issue found, each tagged with its dotted path.
    """
    collector = _IssueCollector()
    result = _to_plain(value, "", collector, projection=projection)
    if not collector.clean():
        raise ConfigError(collector.issues, context="config")
    return result


def to_dict(value: Any) -> Any:
    """Convert a config object into plain JSON-safe Python data.

    Dataclasses become dicts in field-declaration order, ``Enum`` members become
    their values, ``Path`` and ``datetime`` / ``date`` / ``time`` become strings,
    and tuples, sets, and frozensets become lists. Mapping keys pass through these
    same rules. Set and frozenset elements are ordered by their canonical JSON text
    so the output is stable across runs. Constructor fields (``init=True``) form the
    output; ``init=False`` runtime fields are populated by the dataclass lifecycle
    and carried outside serialization.

    The result round-trips: ``from_dict(cls, to_dict(config)) == config`` holds
    for every field whose annotation names a supported type, including bare
    ``tuple``, ``set``, and ``dict``, since ``from_dict`` rebuilds each container
    from its annotation. A field annotated ``Any`` returns in the plain form it was written
    as, so a tuple held in one returns as a list; annotate such a field with a
    container type to restore its exact type.

    Args:
        value: The config object or nested value to convert.

    Returns:
        The converted plain-data structure.

    Raises:
        ConfigError: When a value's type falls outside the supported set and has
          no plain-data form, or holds a non-finite float; the exception lists
          every issue found, each tagged with its dotted path.
    """
    return _project_plain(value, _PlainProjection.EXPORT)


def _to_plain(value: Any, path: str, collector: _IssueCollector, *, projection: _PlainProjection) -> Any:
    """Convert one value to plain data, recording issues with dotted paths.

    Args:
        value: The config object or nested value to convert.
        path: Dotted path of this value, empty at the root.
        collector: Destination for any issues found.
        projection: The field projection carried through every recursive call, so
          a dataclass node emits only the fields the projection selects.

    Returns:
        The converted plain-data structure, or the ``_UNSET`` sentinel when this
        value failed to serialize.
    """
    if is_dataclass(value) and not isinstance(value, type):
        node: dict[str, Any] = {}
        node_failed = False
        for classified in _projected_fields(type(value), projection):
            field_name = classified.definition.name
            item = _to_plain(getattr(value, field_name), _join(path, field_name), collector, projection=projection)
            if item is _UNSET:
                node_failed = True
                continue
            node[field_name] = item
        return _UNSET if node_failed else node
    if isinstance(value, Enum):
        return _to_plain(value.value, path, collector, projection=projection)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.time)):
        # Covers date, datetime (itself a date subclass), and time.
        return value.isoformat()

    array_result = _arrays.array_to_plain(value, path, collector.add)
    if array_result is not _arrays.NOT_ARRAY:
        return _UNSET if array_result is _arrays.FAILED else array_result
    is_numpy_scalar, normalized = _arrays.normalize_numpy_scalar(value)
    if is_numpy_scalar:
        value = normalized

    if isinstance(value, Mapping):
        # Convert keys through the same rules; JSON carries string keys natively.
        mapping: dict[Any, Any] = {}
        mapping_failed = False
        for key, item in value.items():
            item_path = _join(path, str(key))
            plain_key = _to_plain(key, item_path, collector, projection=projection)
            plain_item = _to_plain(item, item_path, collector, projection=projection)
            if plain_key is _UNSET or plain_item is _UNSET:
                mapping_failed = True
                continue
            if isinstance(plain_key, (list, dict)):
                # Keys whose plain form is a container (arrays, tuples) have no
                # mapping-key representation in JSON.
                collector.add(item_path, f"cannot serialize {_typename(key)} as a mapping key")
                mapping_failed = True
                continue
            mapping[plain_key] = plain_item
        return _UNSET if mapping_failed else mapping
    if isinstance(value, (set, frozenset)):
        # Set elements carry the set's own path: iteration order is unstable, so
        # an element index would name a different element on each run.
        elements = [_to_plain(item, path, collector, projection=projection) for item in value]
        if any(element is _UNSET for element in elements):
            return _UNSET
        # Sort by each element's canonical JSON text so the order is total and
        # stable across processes even for mixed-type sets whose elements are not
        # mutually orderable; equal sets must hash equal regardless of PYTHONHASHSEED.
        return sorted(elements, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        items = [
            _to_plain(item, _join(path, str(index)), collector, projection=projection)
            for index, item in enumerate(value)
        ]
        if any(item is _UNSET for item in items):
            return _UNSET
        return items
    if isinstance(value, float):
        if not math.isfinite(value):
            return _reject(collector, path, f"cannot serialize non-finite float {value!r}")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return _reject(collector, path, f"cannot serialize value of type {_typename(value)}")


def _canonical_json(plain: Any) -> str:
    """Encode plain projection data as confingo's canonical compact JSON.

    This single encoding backs both ``config_hash`` and the equality serialized
    fallback, so the two agree token for token: JSON keeps ``true`` / ``1`` /
    ``1.0`` distinct where Python ``==`` would conflate them, and equal configs
    therefore always fingerprint equally.

    Args:
        plain: Plain JSON-safe data from a plain projection.

    Returns:
        Compact JSON with sorted mapping keys.
    """
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Any, *, length: int = 12) -> str:
    """Fingerprint a config with a stable digest over its canonical JSON form.

    The digest ranges over the hashing fields (``init=True``, ``compare=True``,
    effective hash enabled), so a ``compare=False`` or ``hash=False`` field is
    carried by ``to_dict`` yet excluded here. Mapping key order and set iteration
    order are normalized before hashing, so the digest is stable across processes.

    Args:
        config: The config object to fingerprint.
        length: Number of leading hex characters to return, by default 12.

    Returns:
        The truncated SHA-256 digest.
    """
    payload = _canonical_json(_project_plain(config, _PlainProjection.HASH))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
