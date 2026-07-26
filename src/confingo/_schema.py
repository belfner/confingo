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
import os
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
from functools import lru_cache
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

from confingo import _arrays
from confingo._errors import (
    ConfigError,
    ConfigIssue,
    _IssueCollector,
)


_HINT_CACHE: dict[type[Any], dict[str, Any]] = {}

_SCHEMA_CACHE: dict[type[Any], tuple[ConfigIssue, ...]] = {}
"""Per-dataclass cache of schema-validation issues, keyed by the entry type."""

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
    makes the class unhashable; the cache entry is written only after that install
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


class _HintKind(Enum):
    """The unmarshal dispatch category of a resolved type hint."""

    ANY = "any"
    NONE = "none"
    LITERAL = "literal"
    UNION = "union"
    CONTAINER = "container"
    UNSUPPORTED_GENERIC = "unsupported_generic"
    DATACLASS = "dataclass"
    BARE_CONTAINER = "bare_container"
    SCALAR = "scalar"


@dataclass(frozen=True)
class _HintClass:
    """The value-independent structure of one resolved type hint.

    Holds only facts derived from the hint, never a value, path, issue, or
    coercion result, so one instance is reused across every value coerced against
    the hint. Array matching is deliberately excluded: it depends on which
    backends are loaded, which is resolved per operation rather than cached here.

    Attributes:
      kind (_HintKind): The dispatch category.
      stripped (Any): The hint with any ``Annotated`` layers removed.
      origin (Any): ``get_origin(stripped)`` for containers/unions/literals, else None.
      args (tuple[Any, ...]): ``get_args(stripped)`` for containers/unions/literals.
      dataclass_type (type[Any] | None): The dataclass type for a DATACLASS hint.
      bare_origin (Any): The container origin for a BARE_CONTAINER hint.
    """

    kind: _HintKind
    stripped: Any
    origin: Any
    args: tuple[Any, ...]
    dataclass_type: type[Any] | None = None
    bare_origin: Any = None


def _classify_hint_uncached(hint: Any) -> _HintClass:
    """Compute the dispatch structure of a resolved hint without caching.

    Args:
      hint (Any): A resolved type hint, possibly wrapped in ``Annotated``.

    Returns:
      _HintClass: The structural classification, mirroring the unmarshal engine's
        post-strip dispatch order.
    """
    stripped = _strip_annotated(hint)
    if stripped is Any:
        return _HintClass(_HintKind.ANY, stripped, None, ())
    if stripped is type(None):
        return _HintClass(_HintKind.NONE, stripped, None, ())
    origin = get_origin(stripped)
    args = get_args(stripped)
    if origin is Literal:
        return _HintClass(_HintKind.LITERAL, stripped, origin, args)
    if origin is typing.Union or origin is types.UnionType:
        return _HintClass(_HintKind.UNION, stripped, origin, args)
    if origin is not None:
        if origin in _CONTAINER_ORIGINS:
            return _HintClass(_HintKind.CONTAINER, stripped, origin, args)
        return _HintClass(_HintKind.UNSUPPORTED_GENERIC, stripped, origin, args)
    if isinstance(stripped, type):
        if is_dataclass(stripped):
            return _HintClass(_HintKind.DATACLASS, stripped, None, (), dataclass_type=stripped)
        bare = _BARE_CONTAINERS.get(stripped)
        if bare is not None:
            return _HintClass(_HintKind.BARE_CONTAINER, stripped, None, (), bare_origin=bare)
    return _HintClass(_HintKind.SCALAR, stripped, None, ())


_TYPE_CACHE_DISABLED = os.environ.get("CONFINGO_DISABLE_TYPE_CACHE") == "1"
"""Escape hatch: bypass the hint-plan cache for the whole process when set to 1."""

_HINT_PLAN_CACHE_MAX = 2048
"""Bound on distinct compiled hint plans, capping retention of dynamic generics."""


class _IdKey:
    """Identity cache key for a type hint.

    Hashes on ``id(hint)`` and compares by ``is``, so the cache never invokes the
    hint's own ``__hash__`` / ``__eq__`` (safe for unhashable hints such as
    ``Annotated[int, []]`` and for surprising metadata equality). Holding a strong
    reference to the hint for the entry's lifetime prevents an ``id()`` reused
    after garbage collection from aliasing a live entry.
    """

    __slots__ = ("_hash", "hint")

    def __init__(self, hint: Any) -> None:
        self.hint = hint
        self._hash = id(hint)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdKey) and self.hint is other.hint


@lru_cache(maxsize=_HINT_PLAN_CACHE_MAX)
def _classify_hint_by_id(key: _IdKey) -> _HintClass:
    """Cache-backed classification keyed by hint identity.

    Args:
      key (_IdKey): Identity wrapper around the hint.

    Returns:
      _HintClass: The classification for ``key.hint``.
    """
    return _classify_hint_uncached(key.hint)


def _classify_hint(hint: Any) -> _HintClass:
    """Classify a resolved hint, reusing a bounded identity-keyed cache.

    The classification is a pure function of the hint's structure, so it is safe
    to reuse across every value coerced against the hint. Distinct-but-equal hint
    aliases may miss the cache, which is a performance event, never a correctness
    one; each still compiles to an equivalent plan.

    Args:
      hint (Any): A resolved type hint, possibly wrapped in ``Annotated``.

    Returns:
      _HintClass: The structural classification.
    """
    if _TYPE_CACHE_DISABLED:
        return _classify_hint_uncached(hint)
    return _classify_hint_by_id(_IdKey(hint))


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
      config_cls (type[Any]): The entry dataclass to validate.

    Returns:
      tuple[ConfigIssue, ...]: The schema issues found, empty when the schema is fully supported.
    """
    cached = _SCHEMA_CACHE.get(config_cls)
    if cached is not None:
        return cached
    issues: list[ConfigIssue] = []
    entry_message = _entry_type_message(config_cls)
    if entry_message is None:
        _validate_dataclass_schema(config_cls, "", issues, set())
    else:
        # Field classification reads dataclasses.fields, so a non-dataclass entry
        # is reported here rather than walked.
        issues.append(ConfigIssue("", entry_message))
    result = tuple(issues)
    _SCHEMA_CACHE[config_cls] = result
    return result


def _entry_type_message(config_cls: type[Any]) -> str | None:
    """Report why an entry class carries no schema, or None when it is a dataclass.

    Args:
      config_cls (type[Any]): The class an engine operation was entered on.

    Returns:
      str | None: The rejection message when the class is not a dataclass, else None.
    """
    if _is_dataclass_type(config_cls):
        return None
    node_message = _node_message(config_cls)
    if node_message is not None:
        return node_message
    label = getattr(config_cls, "__name__", _typename(config_cls))
    return f"{label} is not a dataclass, so it carries no config schema. Declare it with @dataclass."


def _undecorated_node_message(config_cls: type[Any], hints: Mapping[str, Any]) -> str | None:
    """Report a node subclass whose declaration skipped ``@dataclass``, or None.

    A subclass of a decorated node inherits ``__dataclass_fields__`` through the
    MRO, so ``is_dataclass`` stays true while its own annotations never become
    fields: they load as unknown keys and are absent from every projection.
    Ownership of ``__dataclass_fields__`` is the exact test for whether the
    decorator ran on this class, so ``ClassVar`` and ``InitVar`` annotations --
    which ``fields()`` legitimately omits -- raise no false positive.

    Args:
      config_cls (type[Any]): The dataclass being validated.
      hints (Mapping[str, Any]): The class's resolved annotations.

    Returns:
      str | None: The rejection message when the decorator was skipped on a class
        declaring its own non-``ClassVar`` annotations, else None.
    """
    if "__dataclass_fields__" in config_cls.__dict__:
        return None
    own = config_cls.__dict__.get("__annotations__", {})
    if all(_is_class_var(hints.get(name, Any)) for name in own):
        return None
    return _node_message(config_cls)


def _node_message(config_cls: type[Any]) -> str | None:
    """Describe a ``ConfigNode`` subclass carrying no schema of its own, or None.

    The import is deferred because ``_node`` sits above this module
    (``_node`` -> ``_core`` -> ``_schema``), and it is held here so both callers
    share one cycle-avoiding site.

    Args:
      config_cls (type[Any]): The class being validated.

    Returns:
      str | None: The rejection message when ``config_cls`` is a node, else None.
    """
    from confingo._node import (  # noqa: PLC0415
        _is_config_node,
        _missing_dataclass_message,
    )

    return _missing_dataclass_message(config_cls) if _is_config_node(config_cls) else None


def _is_class_var(hint: Any) -> bool:
    """Report whether a resolved hint declares a ``ClassVar``.

    Args:
      hint (Any): The resolved type hint to inspect.

    Returns:
      bool: True when the hint is ``ClassVar`` or a ``ClassVar[...]`` subscription.
    """
    return hint is ClassVar or get_origin(hint) is ClassVar


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
    undecorated = _undecorated_node_message(config_cls, hints)
    if undecorated is not None:
        issues.append(ConfigIssue(path, undecorated))
    for classified in _classify_dataclass_fields(config_cls).declared:
        field = classified.definition
        field_path = _join(path, field.name)
        issues.extend(ConfigIssue(field_path, message) for message in classified.conflicts)
        if not field.init:
            # init=False fields are runtime state exempt from the supported-type
            # boundary; their annotation need only resolve, checked by
            # _resolved_hints above.
            continue
        before = len(issues)
        _validate_hint_schema(hints[field.name], field_path, issues, seen)
        if len(issues) == before and field.default is not MISSING:
            # The annotation holds up, so the authored default can be judged
            # against it. The value already exists, so reading it runs no
            # user code on this cached path; a default_factory is left to the
            # one build that selects it.
            _validate_direct_default(field.default, hints[field.name], field_path, issues)


def _validate_direct_default(value: Any, hint: Any, path: str, issues: list[ConfigIssue]) -> None:
    """Collect schema issues for one authored ``field(default=...)`` value.

    The import is deferred because ``_defaults`` reads this module's hint
    classification and the serialization walk, both of which sit above it.

    Args:
      value (Any): The authored default object.
      hint (Any): The resolved type hint of the field carrying it.
      path (str): Dotted schema path of the field.
      issues (list[ConfigIssue]): Destination for any schema issues found.
    """
    from confingo._defaults import (  # noqa: PLC0415
        DEFAULT_LABEL,
        validate_authored_value,
    )

    collector = _IssueCollector()
    validate_authored_value(value, hint, path, collector, label=DEFAULT_LABEL)
    issues.extend(collector.issues)


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
        if origin in (set, frozenset) and any(_holds_config_section(arg) for arg in args):
            issues.append(ConfigIssue(path, _section_set_message(hint)))
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


def _holds_config_section(hint: Any) -> bool:
    """Report whether a set element annotation puts a config section where a hash is needed.

    A config dataclass is unhashable, so an element annotation that names one --
    directly, as a union member, or inside the immutable ``tuple`` / ``frozenset``
    shapes that can themselves sit in a set -- describes a collection that cannot
    be built. The walk stops at the dataclass itself: reaching one is the whole
    finding, so nothing recurses into its fields.

    Args:
      hint (Any): The resolved element type hint to inspect.

    Returns:
      bool: True when a config dataclass sits in a hash-bearing position.
    """
    hint = _strip_annotated(hint)
    if _is_dataclass_type(hint):
        return True
    origin = get_origin(hint)
    if origin is typing.Union or origin is types.UnionType or origin in (tuple, frozenset):
        return any(_holds_config_section(arg) for arg in get_args(hint) if arg is not Ellipsis)
    return False


def _section_set_message(hint: Any) -> str:
    """Build the rejection message for a set annotation holding config sections.

    Args:
      hint (Any): The resolved ``set`` / ``frozenset`` hint being rejected.

    Returns:
      str: The rejection message naming the annotation and both remedies.
    """
    return (
        f"config sections are unhashable, so {_hint_name(hint)} cannot be built; use a list or tuple for "
        f"the collection, and use config_hash(section) as the value-identity key when uniqueness matters"
    )


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
