"""Schema introspection over config dataclasses: hints, field metadata, validation.

This module reads the schema surface the marshal / unmarshal engine walks:
resolving a dataclass's annotations to runtime types (installing canonical
equality on first touch), classifying each field for the loading / export /
equality / fingerprint projections, and validating annotations against the
supported type set independently of any config data. Construction
(``_core``) and serialization (``_serialize``) consume these results.
"""

from __future__ import annotations

import datetime as dt
import types
import typing
from collections.abc import (
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
    get_args,
    get_origin,
    get_type_hints,
)

from confingo import _arrays
from confingo._errors import (
    ConfigError,
    ConfigIssue,
)


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
# Type-hint helpers
# ---------------------------------------------------------------------------


def _resolved_hints(config_cls: type[Any]) -> dict[str, Any]:
    """Resolve a dataclass's annotations to runtime type objects, with caching.

    On the first resolution of a class, confingo installs canonical equality and
    identity hashing on it; the cache entry is written only after that install
    succeeds, so a class that violates the ownership contract is re-checked and
    re-rejected on every touch.

    Args:
      config_cls (type[Any]): The dataclass whose annotations to resolve.

    Returns:
      dict[str, Any]: Mapping of field name to resolved type hint, with
        ``Annotated`` metadata preserved (``include_extras=True``) so array
        dtype and shape metadata survive for the array classifier; consumers
        strip it after array classification.

    Raises:
      ConfigError: When an annotation names a type that is unreachable from the
        defining module's namespace, or the class defines its own ``__eq__`` /
        ``__hash__`` or a conflicting ``@dataclass`` flag. Declare config
        dataclasses at module level so every name they reference resolves there.
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
    from confingo._equality import _install_canonical_eq  # noqa: PLC0415

    _install_canonical_eq(config_cls)
    _HINT_CACHE[config_cls] = hints
    return hints


def _strip_annotated(hint: Any) -> Any:
    """Unwrap ``Annotated`` layers, returning the underlying type hint.

    Args:
      hint (Any): A resolved type hint, possibly wrapped in ``Annotated``.

    Returns:
      Any: The underlying hint; hints without metadata return unchanged.
    """
    while get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return hint


def _is_dataclass_type(hint: Any) -> bool:
    """Report whether a type hint is a dataclass type.

    Args:
      hint (Any): The resolved type hint to inspect.

    Returns:
      bool: True when the hint is a dataclass type rather than an instance.
    """
    return isinstance(hint, type) and is_dataclass(hint)


@dataclass(frozen=True)
class _ClassifiedField:
    """One dataclass field with the facts the engine derives once per field.

    Attributes:
      definition (Field[Any]): The underlying ``dataclasses.Field``. Projection
        membership (``init``, ``compare``, ``hash``) is read from it directly.
      has_default (bool): Whether the field carries a default or default_factory.
      conflicts (tuple[str, ...]): Contradictory-flag messages for the field,
        empty when valid.
    """

    definition: Field[Any]
    has_default: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class _DataclassFields:
    """A dataclass's fields grouped by projection, computed once and cached.

    Attributes:
      declared (tuple[_ClassifiedField, ...]): Every field in declaration order.
      init_fields (tuple[_ClassifiedField, ...]): Fields built from config input
        and emitted by ``to_dict`` (``init=True``).
      non_init (tuple[_ClassifiedField, ...]): Fields excluded from construction
        (``init=False``), populated in ``__post_init__`` and checked for
        completeness after construction.
      compared (tuple[_ClassifiedField, ...]): Fields equality includes
        (``init and compare``).
      hashed (tuple[_ClassifiedField, ...]): Fields ``config_hash`` includes
        (``init and compare and effective_hash``).
      by_name (Mapping[str, _ClassifiedField]): Every declared field keyed by name.
    """

    declared: tuple[_ClassifiedField, ...]
    init_fields: tuple[_ClassifiedField, ...]
    non_init: tuple[_ClassifiedField, ...]
    compared: tuple[_ClassifiedField, ...]
    hashed: tuple[_ClassifiedField, ...]
    by_name: Mapping[str, _ClassifiedField]


_FIELD_CACHE: dict[type[Any], _DataclassFields] = {}
"""Per-dataclass cache of field classifications, keyed by class identity."""


def _field_hashed(definition: Field[Any]) -> bool:
    """Report whether ``config_hash`` includes a field.

    ``init`` is the master switch; on an ``init=True`` field ``compare`` scopes
    equality and the effective hash (``hash`` when set, else ``compare``) scopes
    the fingerprint.

    Args:
      definition (Field[Any]): The dataclass field to test.

    Returns:
      bool: True when the field participates in the fingerprint.
    """
    effective_hash = definition.compare if definition.hash is None else definition.hash
    return definition.init and definition.compare and effective_hash


def _classify_dataclass_fields(config_cls: type[Any]) -> _DataclassFields:
    """Classify a dataclass's fields for every engine projection, with caching.

    ``init`` is the master switch: an ``init=False`` field is excluded from
    loading, export, equality, and the fingerprint, so its ``compare`` / ``hash``
    flags are inert. On an ``init=True`` field ``compare`` scopes equality (and
    therefore the fingerprint) and ``hash`` scopes the fingerprint, with the one
    contradiction ``hash=True, compare=False`` recorded in ``conflicts``.

    Args:
      config_cls (type[Any]): The dataclass to classify.

    Returns:
      _DataclassFields: The grouped classification, cached by class identity.
    """
    cached = _FIELD_CACHE.get(config_cls)
    if cached is not None:
        return cached
    declared: list[_ClassifiedField] = []
    for field in fields(config_cls):
        conflicts: list[str] = []
        if field.init and field.hash is True and field.compare is False:
            conflicts.append(
                "field(hash=True, compare=False) is contradictory: config_hash fields must participate in equality"
            )
        declared.append(
            _ClassifiedField(
                definition=field,
                has_default=field.default is not MISSING or field.default_factory is not MISSING,
                conflicts=tuple(conflicts),
            )
        )
    result = _DataclassFields(
        declared=tuple(declared),
        init_fields=tuple(item for item in declared if item.definition.init),
        non_init=tuple(item for item in declared if not item.definition.init),
        compared=tuple(item for item in declared if item.definition.init and item.definition.compare),
        hashed=tuple(item for item in declared if _field_hashed(item.definition)),
        by_name={item.definition.name: item for item in declared},
    )
    _FIELD_CACHE[config_cls] = result
    return result


def _validate_schema(config_cls: type[Any]) -> tuple[ConfigIssue, ...]:
    """Validate a dataclass's field annotations against the supported type set.

    This inspects the schema itself, independent of any config data, so an
    unsupported annotation is reported even when the field is omitted and falls
    back to its default. Field default values are left untouched.

    Args:
      config_cls (type[Any]): The root dataclass to validate.

    Returns:
      tuple[ConfigIssue, ...]: The schema issues found, empty when the schema is fully supported.
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
      config_cls (type[Any]): The dataclass to inspect.
      path (str): Dotted schema path of this node, empty at the root.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      seen (set[type[Any]]): Dataclasses already visited on this path, to break reference cycles.
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
      hint (Any): The resolved type hint to inspect.
      path (str): Dotted schema path of the field carrying this hint.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      seen (set[type[Any]]): Dataclasses already visited on this path, to break reference cycles.
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
      hint (Any): The resolved type hint to describe.

    Returns:
      str: A display name such as ``int``, ``str | None``, or ``list[Path]``.
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
      value (Any): The value to describe.

    Returns:
      str: The type name, with ``None`` reported as ``None``.
    """
    return "None" if value is None else type(value).__name__


def _join(path: str, key: str) -> str:
    """Append a key to a dotted config path.

    Args:
      path (str): The parent dotted path, empty at the root.
      key (str): The child key or index to append.

    Returns:
      str: The combined dotted path.
    """
    return key if path == "" else f"{path}.{key}"
