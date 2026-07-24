"""Dataclass construction from plain data: ``from_dict`` and value coercion.

``from_dict`` builds a dataclass tree from a nested mapping, coercing each value
toward its annotated type and collecting every problem before raising. Schema
introspection lives in ``_schema``; serialization and the fingerprint live in
``_serialize``; this module is the unmarshal half of the engine.

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
"""

from __future__ import annotations

import datetime as dt
import math
import types
import typing
from collections.abc import (
    Mapping,
    Sequence,
)
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    get_args,
    get_origin,
)

from confingo import _arrays
from confingo._errors import (
    _UNSET,
    ConfigError,
    _IssueCollector,
    _reject,
)
from confingo._schema import (
    _BARE_CONTAINERS,
    _CONTAINER_ORIGINS,
    _SEQUENCE_BUILDERS,
    _classify_dataclass_fields,
    _hint_name,
    _is_dataclass_type,
    _join,
    _resolved_hints,
    _strip_annotated,
    _typename,
    _validate_schema,
)


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

T = TypeVar("T")


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
      config_cls (type[T]): The root dataclass to build.
      data (Mapping[str, Any]): Nested mapping of config values, typically parsed
        from a config file.
      context (str = "config"): Description of the config source used in the error
        summary.

    Returns:
      T: The constructed config object.

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
      config_cls (type[Any]): The dataclass to construct.
      data (Any): The mapping of values for this node.
      path (str): Dotted path of this node, empty at the root.
      collector (_IssueCollector): Destination for any issues found.
      implicit_chain (tuple[type[Any], ...] = ()): Dataclass types currently being
        built implicitly on this branch, used to terminate self-referential
        schemas.

    Returns:
      Any: The constructed instance, or the ``_UNSET`` sentinel when this node
        failed to build.
    """
    if not isinstance(data, Mapping):
        return _reject(collector, path, f"expected a mapping for {config_cls.__name__}, got {_typename(data)}")

    hints = _resolved_hints(config_cls)
    classification = _classify_dataclass_fields(config_cls)
    loadable_names = {item.definition.name for item in classification.init_fields}
    for key in data:
        if key in classification.by_name and key not in loadable_names:
            collector.add(_join(path, str(key)), "field is not configurable (init=False)")
        elif key not in loadable_names:
            collector.add(_join(path, str(key)), f"unknown key (known keys: {', '.join(sorted(loadable_names))})")

    kwargs: dict[str, Any] = {}
    node_failed = False
    for classified in classification.init_fields:
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
      value (Any): The raw value from the config mapping.
      hint (Any): The resolved type hint the value must satisfy.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The coerced value, or the ``_UNSET`` sentinel when coercion failed.
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
      value (Any): The raw value from the config mapping.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The value unchanged, or the ``_UNSET`` sentinel when it holds a
        non-finite float.
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
      value (Any): The raw value from the config mapping.
      hint (Any): The union type hint, used for the error message.
      args (tuple[Any, ...]): The union's member types.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The coerced value, or the ``_UNSET`` sentinel when no member matched.
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
      items (list[Any]): The raw elements to coerce.
      element_hints (list[Any]): The type hint for each element, aligned by index
        with ``items``.
      path (str): Dotted path of the container holding these items.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The list of coerced items, or the ``_UNSET`` sentinel when any element
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
      value (Any): The raw value from the config mapping.
      hint (Any): The container type hint, used for the error message.
      origin (Any): The container's unsubscripted origin type.
      args (tuple[Any, ...]): The container's element type arguments.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The coerced container, or the ``_UNSET`` sentinel when coercion failed.
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

    A short router over the supported leaf types: it dispatches to the enum,
    numeric, string, path, and ISO temporal branches in that fixed order (with
    ``datetime`` decided before ``date``, since ``datetime`` subclasses it).

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
        return _coerce_enum(value, hint, path, collector)
    if hint in (bool, int, float):
        return _coerce_numeric(value, hint, path, collector)

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
        return _coerce_iso(value, dt.datetime.fromisoformat, "datetime", path, collector)

    if issubclass(hint, dt.date):
        # datetime is a subclass of date and is handled above, so a datetime on a
        # plain date field is a type mismatch rather than a silent truncation.
        if isinstance(value, dt.datetime):
            return _reject(collector, path, f"expected a date, got {_typename(value)}")
        if isinstance(value, dt.date):
            return value
        return _coerce_iso(value, dt.date.fromisoformat, "date", path, collector)

    if issubclass(hint, dt.time):
        if isinstance(value, dt.time):
            return value
        return _coerce_iso(value, dt.time.fromisoformat, "time", path, collector)

    return _reject(collector, path, f"unsupported field type {hint.__name__}")


def _coerce_enum(value: Any, hint: type[Enum], path: str, collector: _IssueCollector) -> Any:
    """Resolve a value to an ``Enum`` member by value, then by member name.

    Args:
        value: The raw value from the config mapping.
        hint: The ``Enum`` subclass to resolve against.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The resolved enum member, or the ``_UNSET`` sentinel when no member matched.
    """
    try:
        return hint(value)
    except ValueError:
        pass
    if isinstance(value, str) and value in hint.__members__:
        return hint[value]
    options = ", ".join(repr(member.value) for member in hint)
    return _reject(collector, path, f"expected one of {options} for enum {hint.__name__}, got {value!r}")


def _coerce_numeric(value: Any, hint: type, path: str, collector: _IssueCollector) -> Any:
    """Coerce a value toward ``bool``, ``int``, or ``float``.

    ``bool`` is an ``int`` subclass, so it is kept off ``int`` / ``float`` fields
    explicitly, and whole-number floats land on ``int`` fields so forms like
    ``1e6`` are accepted.

    Args:
        value: The raw value from the config mapping.
        hint: One of ``bool``, ``int``, or ``float``.
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The coerced number, or the ``_UNSET`` sentinel when coercion failed.
    """
    if hint is bool:
        if isinstance(value, bool):
            return value
        return _reject(collector, path, f"expected bool, got {_typename(value)}")

    if hint is int:
        if isinstance(value, bool):
            return _reject(collector, path, f"expected int, got {_typename(value)}")
        if isinstance(value, int):
            return value
        # Forms like 1e6 parse as float; accept whole-number floats onto int fields.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return _reject(collector, path, f"expected int, got {_typename(value)}")

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


def _coerce_iso(
    value: Any,
    parser: Callable[[str], Any],
    label: str,
    path: str,
    collector: _IssueCollector,
) -> Any:
    """Parse an ISO 8601 string into a temporal value, or reject.

    Called after the value is known not to be an acceptable temporal instance, so
    only a string can succeed. The base ``datetime`` / ``date`` / ``time``
    parsers are passed in and only ``ValueError`` is caught, so subclass hints
    keep the standard-library parsing behavior.

    Args:
        value: The raw value from the config mapping.
        parser: The ``fromisoformat`` parser for the target temporal type.
        label: The type name used in the issue message (``datetime`` / ``date`` /
          ``time``).
        path: Dotted path of this value.
        collector: Destination for any issues found.

    Returns:
        The parsed temporal value, or the ``_UNSET`` sentinel when the value was
        not an ISO 8601 string of the expected form.
    """
    if isinstance(value, str):
        try:
            return parser(value)
        except ValueError:
            return _reject(collector, path, f"expected an ISO 8601 {label} string, got {value!r}")
    return _reject(collector, path, f"expected an ISO 8601 {label} string, got {_typename(value)}")
