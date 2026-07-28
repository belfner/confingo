"""Dataclass construction from plain data: ``from_dict`` and value coercion.

``from_dict`` builds a dataclass tree from a nested mapping, coercing each value
toward its annotated type and collecting every problem before raising. Schema
introspection lives in ``_schema``; serialization and the fingerprint live in
``_serialize``; this module is the unmarshal half of the engine.

Runs on Python 3.12 and newer, so generics use PEP 695 syntax throughout.

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
author wrote it. Defaults are validated against their annotation and their plain
form, then used exactly as authored, so a default already carries the runtime
type the annotation names. A dataclass-typed field with no default builds
implicitly from an empty mapping, recursively, so its own required values are
reported at their nested dotted paths. Every other field with no default is
required.

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
import typing
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import MISSING
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    get_origin,
)

from confingo import _arrays
from confingo._defaults import (
    DEFAULT_LABEL,
    FACTORY_LABEL,
    validate_authored_value,
)
from confingo._errors import (
    _UNSET,
    RESOURCE_ERRORS,
    ConfigError,
    _IssueCollector,
    _reject,
    class_label,
)
from confingo._schema import (
    _SEQUENCE_BUILDERS,
    MAX_PLAIN_DEPTH,
    _class_namespaces,
    _ClassifiedField,
    _classify_dataclass_fields,
    _classify_hint,
    _hint_name,
    _HintKind,
    _is_dataclass_type,
    _join,
    _resolved_hints,
    _strip_annotated,
    _typename,
    _validate_schema,
    array_match,
    plain_cycle_message,
    plain_data_message,
    plain_depth_message,
    plain_key_message,
    plain_scalar_message,
    unsupported_hint_message,
)
from confingo.typing import ConfigScalar


if TYPE_CHECKING:
    from collections.abc import Callable


def validate_schema(config_cls: type[Any], *, context: str = "config schema") -> None:
    """Check a config dataclass's schema without building anything from it.

    Walks the whole tree the class declares, recursing into nested dataclasses
    and into dataclasses held in lists, tuples, sets, and dict values, and
    reports every annotation outside the supported set along with every authored
    ``field(default=...)`` that does not already carry its annotation's runtime
    type and a plain serializable form. No config data is read and no
    ``default_factory`` is called, so this answers whether the schema is
    well formed before any file exists.

    This is the same check ``from_dict`` runs before it builds, so a class that
    validates here raises no schema issue at load time.

    Args:
      config_cls (type[Any]): The entry dataclass to validate.
      context (str = "config schema"): Description used in the error summary.

    Raises:
      ConfigError: When the schema carries any issue. The exception lists every
        issue found in the whole tree, each tagged with its dotted path.
    """
    issues = _validate_schema(config_cls)
    if len(issues) > 0:
        raise ConfigError(issues, context=context)


def from_dict[T](config_cls: type[T], data: Mapping[str, Any], *, context: str = "config") -> T:
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

    Authored defaults are validated rather than coerced. A direct
    ``field(default=...)`` value is checked during schema preflight, so a wrong
    default is reported whether or not the input supplies the field. A selected
    ``default_factory`` runs once here, and its one product is validated and
    passed on unchanged. Both gates require the value to already carry the
    annotation's runtime type and to have a plain serializable form.

    An ``init=False`` field is runtime state: it is not built from the mapping
    (supplying its key is reported as not configurable), and it is populated by
    its default or in ``__post_init__``. After each node is constructed, every
    ``init=False`` field is checked for population, so one left unset by
    ``__post_init__`` is reported rather than surfacing later as an
    ``AttributeError``.

    Args:
      config_cls (type[T]): The entry dataclass to build.
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
        raise ConfigError(collector.issues, context=context, pending_lifecycle_paths=collector.pending_lifecycle_paths)
    return typing.cast("T", instance)


def _build(
    config_cls: type[Any],
    data: Any,
    path: str,
    collector: _IssueCollector,
    implicit_chain: tuple[type[Any], ...] = (),
    data_chain: tuple[int, ...] = (),
    depth: int = 0,
) -> Any:
    """Construct one dataclass node, recording issues rather than raising.

    A section is the one place the build walk can descend without an annotation
    bounding it, since a recursive schema reads as many sections as the supplied
    mapping nests. ``data_chain`` bounds that: a mapping already open on this
    branch reaches itself, and a chain as long as ``MAX_PLAIN_DEPTH`` is deeper
    than the walk supports. Both arrive as an issue at the path they reach.

    Args:
      config_cls (type[Any]): The dataclass to construct.
      data (Any): The mapping of values for this node.
      path (str): Dotted path of this node, empty at the root.
      collector (_IssueCollector): Destination for any issues found.
      implicit_chain (tuple[type[Any], ...] = ()): Dataclass types currently being
        built implicitly on this branch, used to terminate self-referential
        schemas.
      data_chain (tuple[int, ...] = ()): Ids of the supplied mappings open on this
        branch, ending a mapping that reaches itself.
      depth (int = 0): How deep in the plain document this node sits, counting the
        entry mapping as level zero.

    Returns:
      Any: The constructed instance, or the ``_UNSET`` sentinel when this node
        failed to build.
    """
    # Each of these three ends the node before its fields are read, so the whole
    # lifecycle is still ahead of it and the node records every stage it declares.
    if not isinstance(data, Mapping):
        _record_pending_lifecycle(config_cls, path, collector)
        return _reject(collector, path, f"expected a mapping for {class_label(config_cls)}, got {_typename(data)}")
    if depth >= MAX_PLAIN_DEPTH:
        _record_pending_lifecycle(config_cls, path, collector)
        return _reject(collector, path, plain_depth_message())
    if id(data) in data_chain:
        _record_pending_lifecycle(config_cls, path, collector)
        return _reject(collector, path, plain_cycle_message())
    node_chain = (*data_chain, id(data))

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
            supplied = _absent_field_value(
                classified,
                hints[field.name],
                field_path,
                collector,
                implicit_chain=implicit_chain,
                data_chain=node_chain,
                depth=depth + 1,
            )
            if supplied is _UNSET:
                node_failed = True
            elif supplied is not _KEEP_DECLARED:
                kwargs[field.name] = supplied
            continue
        coerced = _coerce(
            data[field.name], hints[field.name], field_path, collector, data_chain=node_chain, depth=depth + 1
        )
        if coerced is _UNSET:
            node_failed = True
            continue
        kwargs[field.name] = coerced

    # Walk every field before bailing so one build surfaces all of a node's
    # problems at once; kwargs would be incomplete once any field failed.
    if node_failed:
        _record_pending_lifecycle(config_cls, path, collector)
        return _UNSET

    instance = _run_callback(lambda: config_cls(**kwargs), "constructing", config_cls, path, collector)
    if instance is _UNSET:
        # The call names one step from the outside, so a constructor that raised
        # before reaching ``__post_init__`` reads the same here as one that raised
        # inside it. ``__post_init__`` stays among the stages ahead for that reason:
        # a hook that already raised runs again once the issue it reported is fixed.
        _record_pending_lifecycle(config_cls, path, collector)
        return _UNSET

    if not _check_node_lifecycle(instance, config_cls, path, collector):
        # Construction and the completeness check are behind it; ``__validate__``
        # is what the next load reaches.
        _record_pending_lifecycle(config_cls, path, collector, post_init=False, completeness=False)
        return _UNSET
    return instance


_NO_BINDING = object()
"""Sentinel answering a lookup that reached the end of a class's MRO."""


def _nearest_binding(config_cls: type[Any], name: str) -> Any:
    """Read what a name binds to on a class, taking the most derived namespace.

    The raw namespaces themselves answer, whatever a metaclass supplies under
    ``__getattribute__``, matching how the rest of the schema machinery inspects a
    class. Taking the nearest binding is what makes
    a derived ``__validate__ = None`` answer for the base method it shadows.

    Args:
      config_cls (type[Any]): The class to inspect.
      name (str): The attribute name to resolve.

    Returns:
      Any: The binding found, or the ``_NO_BINDING`` sentinel.
    """
    for namespace in _class_namespaces(config_cls):
        if name in namespace:
            return namespace[name]
    return _NO_BINDING


def _declares_hook(config_cls: type[Any], name: str) -> bool:
    """Report whether a class declares a lifecycle hook a later load would run.

    A hook reaches the engine as a callable read off an instance, so a callable
    binding counts and so does a descriptor, which supplies its callable once an
    instance exists. Any other binding, ``None`` among them, is the shadow it looks
    like, matching the ``callable`` test the lifecycle itself applies.

    The descriptor reading goes through the raw namespaces of the binding's own
    type for the same reason the binding itself does: that type belongs to the
    config author too, so an ordinary attribute read there would run a metaclass
    ``__getattribute__`` and let a report about a config be replaced by a failure
    inside the machinery describing it.

    Args:
      config_cls (type[Any]): The class to inspect.
      name (str): The hook name, ``__post_init__`` or ``__validate__``.

    Returns:
      bool: True when a later load would find a hook of that name.
    """
    binding = _nearest_binding(config_cls, name)
    if binding is _NO_BINDING:
        return False
    if callable(binding):
        return True
    return _nearest_binding(type(binding), "__get__") is not _NO_BINDING


def _record_pending_lifecycle(
    config_cls: type[Any],
    path: str,
    collector: _IssueCollector,
    *,
    post_init: bool = True,
    completeness: bool = True,
    validate: bool = True,
) -> None:
    """Record a node whose remaining lifecycle stages run on a later load.

    The flags name the stages still ahead of the node at the point it was set
    aside, so a node that failed after construction carries fewer of them than one
    that stopped ahead of its constructor.

    Args:
      config_cls (type[Any]): The class whose declarations decide the stages.
      path (str): Dotted path of the node.
      collector (_IssueCollector): Destination for the pending path.
      post_init (bool = True): Whether ``__post_init__`` is still ahead.
      completeness (bool = True): Whether the ``init=False`` check is still ahead.
      validate (bool = True): Whether ``__validate__`` is still ahead.
    """
    if post_init and _declares_hook(config_cls, "__post_init__"):
        collector.add_pending_lifecycle(path)
        return
    if completeness and len(_classify_dataclass_fields(config_cls).non_init) > 0:
        collector.add_pending_lifecycle(path)
        return
    if validate and _declares_hook(config_cls, "__validate__"):
        collector.add_pending_lifecycle(path)


def _holds_sections(value: Any, hint: Any = None) -> bool:
    """Report whether a barrier set aside something whose lifecycle a later load can run.

    Read at an authored-default barrier, where a product is put down with its
    lifecycle unvisited. Two readings answer, since each covers a case the other
    leaves open. The product answers for what is there: a section carries its own
    hooks and a container may hold sections anywhere inside it, so a container
    answers True on shape alone. The annotation answers for what a repaired default
    builds: a product rejected against a section annotation is a leaf precisely
    because it failed, and the section the annotation names reaches its hooks once
    the default is fixed. The annotation is also the whole reading when a factory
    raised, where the product it would have made stays unbuilt.

    A container or union annotation whose repaired default carries leaves alone
    answers True here, naming a path that stays quiet. That direction is
    deliberate: naming a path that turns out quiet costs a reader one entry, and
    staying silent about a path that later reports is the surprise this whole
    signal exists to end.

    Args:
      value (Any): The product the barrier set aside.
      hint (Any = None): The annotation the field declares, when one is in hand.

    Returns:
      bool: True when lifecycle work may remain at or beneath the path.
    """
    if not isinstance(value, type) and _is_dataclass_type(type(value)):
        return True
    # ``Sequence`` covers the concrete builders and every other sequence a default
    # may hold, each able to carry sections. ``str`` and ``bytes`` are sequences of
    # themselves and hold leaves, which container coercion reads the same way.
    if isinstance(value, (Mapping, set, frozenset)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return True
    if hint is None:
        return False
    kind = _classify_hint(hint).kind
    return kind in (_HintKind.DATACLASS, _HintKind.CONTAINER, _HintKind.UNION)


def _check_node_lifecycle(instance: Any, config_cls: type[Any], path: str, collector: _IssueCollector) -> bool:
    """Check one constructed node for completeness, then report what it declares.

    Every ``init=False`` field must be populated by its default or by
    ``__post_init__`` before ``__validate__`` or user code reads it.
    ``object.__getattribute__`` probes the real attribute, bypassing any
    ``__getattr__`` fallback, for both ordinary and slots classes. ``__validate__``
    runs only on a fully populated instance, so a node still awaiting a runtime
    value contributes that completeness issue in its place.

    Args:
      instance (Any): The constructed config object.
      config_cls (type[Any]): The class the instance belongs to.
      path (str): Dotted path of this node.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      bool: True when the node was fully populated, whatever ``__validate__``
        went on to report.
    """
    node_incomplete = False
    for classified in _classify_dataclass_fields(config_cls).non_init:
        name = classified.definition.name
        try:
            object.__getattribute__(instance, name)
        except AttributeError:
            collector.add(_join(path, name), "init=False field was not set during __post_init__")
            node_incomplete = True
    if node_incomplete:
        return False

    validate = _run_callback(partial(_lookup_hook, instance), "validating", config_cls, path, collector)
    if validate is not _UNSET and callable(validate):
        _collect_validate_messages(validate, config_cls, path, collector)
    return True


def _check_authored_sections(value: Any, path: str, collector: _IssueCollector, seen: tuple[int, ...] = ()) -> None:
    """Run the section lifecycle over every config object an authored default holds.

    An authored default arrives already constructed, so its ``__post_init__`` has
    run inside the factory or the schema declaration and nothing else has. A
    section built from supplied data is checked for ``init=False`` completeness
    and then reports what ``__validate__`` returns, so the same section reaches
    the same report whether the input carried it or the default supplied it.

    The walk reaches sections through the containers a default can hold, and an
    identity stack ends a structure that reaches itself. The value's plain form
    and runtime type are settled before this runs, so nesting is already bounded.

    Args:
      value (Any): The authored value, or one of the values it holds.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      seen (tuple[int, ...] = ()): Ids of the values open on this branch.
    """
    if id(value) in seen:
        return
    branch = (*seen, id(value))
    # A section is read as a section first, matching the order the marshal walk
    # tests in. Preflight rejects a class that is both a section and a container,
    # so the two orders agree on every schema that gets this far; testing the same
    # way keeps that agreement a property of the code rather than of the rule.
    if not isinstance(value, type) and _is_dataclass_type(type(value)):
        section_cls = type(value)
        if not _check_node_lifecycle(value, section_cls, path, collector):
            # The walk stops here, so this section's ``__validate__`` and every
            # section it holds stay unvisited. One entry names the path the walk
            # put down, covering what sits beneath it.
            collector.add_pending_lifecycle(path)
            return
        for classified in _classify_dataclass_fields(section_cls).init_fields:
            name = classified.definition.name
            _check_authored_sections(getattr(value, name, None), _join(path, name), collector, branch)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_authored_sections(item, _join(path, str(key)), collector, branch)
        return
    if isinstance(value, (set, frozenset)):
        # Set iteration order is unstable, so an element index would name a
        # different element on each run; elements carry the set's own path.
        for item in value:
            _check_authored_sections(item, path, collector, branch)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_authored_sections(item, _join(path, str(index)), collector, branch)


def _lookup_hook(instance: Any) -> Any:
    """Read the ``__validate__`` hook from one instance.

    The read is a call into the instance's own class, which may define
    ``__getattribute__`` or ``__getattr__``, so it runs inside the callback guard
    alongside the hook it looks up.

    Args:
      instance (Any): The constructed config object.

    Returns:
      Any: The bound hook, or None when the class declares none.
    """
    return getattr(instance, "__validate__", None)


def _run_callback(
    call: Callable[[], Any], verb: str, config_cls: type[Any], path: str, collector: _IssueCollector
) -> Any:
    """Run one user callback, turning what it raises into an issue.

    A callback belongs to the config author, so anything it raises describes the
    config rather than confingo, and it arrives as one issue beside the issues
    already collected. ``MemoryError`` and ``SystemError`` describe the
    interpreter's own state instead, so they travel on unchanged, as does any
    ``BaseException`` outside ``Exception``.

    Args:
      call (Callable[[], Any]): The callback, already bound to its arguments.
      verb (str): What the call was doing, naming it in the message.
      config_cls (type[Any]): The class whose callback this is.
      path (str): Dotted path of the node.
      collector (_IssueCollector): Destination for any issue found.

    Returns:
      Any: Whatever the callback returned, or the ``_UNSET`` sentinel when it raised.
    """
    try:
        return call()
    except RESOURCE_ERRORS:
        raise
    except Exception as exc:
        return _reject(
            collector, path, f"{verb} {class_label(config_cls)} raised {_exception_label(exc)}: {_describe(exc)}"
        )


def _exception_label(exc: BaseException) -> str:
    """Name an exception's class without letting the reading fail again.

    Reading ``type(exc).__name__`` is a metaclass attribute read, which a class
    that defines one can answer with code of its own. A class that raises there
    would replace the callback failure being reported with a failure of its own,
    so a fixed phrase answers instead and the original failure still arrives.

    Args:
      exc (BaseException): The caught exception.

    Returns:
      str: The exception's class name, or a fixed phrase when reading it fails.
    """
    try:
        return str.__str__(type(exc).__name__)
    except RESOURCE_ERRORS:
        raise
    except Exception:
        return "an exception whose class could not be named"


_HOOK_CONTRACT_REMEDY = (
    "__validate__ returns an iterable of messages; return a list of strings, or an empty list when the config is valid"
)
"""What ``__validate__`` must hand back, named wherever its return is rejected."""


def _collect_validate_messages(
    validate: Callable[[], Any], config_cls: type[Any], path: str, collector: _IssueCollector
) -> None:
    """Run ``__validate__`` and collect the messages it reported.

    The return is read against the documented contract before it is consumed. A
    ``str`` satisfies "an iterable" while iterating to one issue per character,
    and ``None`` is what an ``if bad: return [...]`` with no trailing return hands
    back; each is reported as the contract issue it is rather than acted on.

    Args:
      validate (Callable[[], Any]): The bound ``__validate__`` method.
      config_cls (type[Any]): The class whose hook this is.
      path (str): Dotted path of the node.
      collector (_IssueCollector): Destination for any issues found.
    """
    returned = _run_callback(validate, "validating", config_cls, path, collector)
    if returned is _UNSET:
        return
    if isinstance(returned, (str, bytes)) or not isinstance(returned, Iterable):
        described = _run_callback(partial(_render_type, returned), "validating", config_cls, path, collector)
        if described is _UNSET:
            return
        collector.add(path, f"{class_label(config_cls)}.__validate__ returned {described}; {_HOOK_CONTRACT_REMEDY}")
        return
    messages = _run_callback(
        lambda: list(typing.cast("Iterable[Any]", returned)), "validating", config_cls, path, collector
    )
    if messages is _UNSET:
        return
    for message in messages:
        rendered = _run_callback(partial(_render_message, message), "validating", config_cls, path, collector)
        if rendered is not _UNSET:
            collector.add(path, rendered)


def _render_type(value: Any) -> str:
    """Name the type of what ``__validate__`` handed back.

    Args:
      value (Any): The returned object.

    Returns:
      str: The type's display name.
    """
    return _typename(value)


def _render_message(message: Any) -> str:
    """Render one reported message as text.

    Args:
      message (Any): One item from what ``__validate__`` returned.

    Returns:
      str: The item's text.
    """
    return str(message)


_KEEP_DECLARED = object()
"""Sentinel meaning a field is left out of ``kwargs`` so its declaration applies."""


def _absent_field_value(
    classified: _ClassifiedField,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    *,
    implicit_chain: tuple[type[Any], ...],
    data_chain: tuple[int, ...],
    depth: int,
) -> Any:
    """Decide what an ``init=True`` field the input omitted contributes to ``kwargs``.

    Args:
      classified (_ClassifiedField): The omitted field's classification.
      hint (Any): The field's resolved type hint.
      path (str): Dotted path of the field.
      collector (_IssueCollector): Destination for any issues found.
      implicit_chain (tuple[type[Any], ...]): Dataclass types currently being built
        implicitly on this branch, used to terminate self-referential schemas.
      data_chain (tuple[int, ...]): Ids of the supplied mappings open on this branch.
      depth (int): How deep in the plain document this field's value sits.

    Returns:
      Any: The value to pass in ``kwargs``, the ``_KEEP_DECLARED`` sentinel when
        the field's own default applies, or the ``_UNSET`` sentinel when the field
        failed.
    """
    field = classified.definition
    if field.default_factory is not MISSING:
        # The factory is selected, so it runs exactly once here and its one product
        # is validated and passed on; leaving it out of kwargs would let the
        # constructor call it a second time and store a value nothing checked.
        return _selected_factory_value(field.default_factory, hint, path, collector, depth)
    if classified.has_default:
        # A direct default is validated during schema preflight, where its type and
        # its plain form are settled once. What preflight cannot know is where the
        # field sits at the build that selects the default, since a section reached
        # through a container or through its own recursion sits at a runtime level
        # the schema path does not name. The value's own depth was measured with
        # the rest of its class, so the budget is spent here without walking it.
        if depth + classified.default_depth > MAX_PLAIN_DEPTH:
            # The budget runs out before the lifecycle walk reaches the authored
            # value, so whatever sections it holds stay unvisited.
            if _holds_sections(field.default, hint):
                collector.add_pending_lifecycle(path)
            return _reject(collector, path, f"{DEFAULT_LABEL}: {plain_depth_message()}")
        _check_authored_sections(field.default, path, collector)
        return _KEEP_DECLARED
    stripped = _strip_annotated(hint)
    if not _is_dataclass_type(stripped):
        return _reject(collector, path, "missing required value")
    # An absent dataclass section builds implicitly from an empty mapping, so its
    # own required leaves surface at their nested paths. The chain terminates
    # self-referential schemas.
    if stripped in implicit_chain:
        return _reject(collector, path, "missing required value")
    return _build(stripped, {}, path, collector, (*implicit_chain, stripped), data_chain, depth)


def _selected_factory_value(
    factory: Callable[[], Any], hint: Any, path: str, collector: _IssueCollector, depth: int
) -> Any:
    """Run a selected ``default_factory`` once and validate the value it produced.

    Args:
      factory (Callable[[], Any]): The selected factory.
      hint (Any): The resolved type hint the produced value must already carry.
      path (str): Dotted path of the field.
      collector (_IssueCollector): Destination for any issues found.
      depth (int): How deep in the plain document the produced value sits.

    Returns:
      Any: The produced object, unchanged, or the ``_UNSET`` sentinel when the
        factory raised or its value failed validation.
    """
    try:
        produced = factory()
    except RESOURCE_ERRORS:
        raise
    except Exception as exc:
        # The factory belongs to the config author, so whatever it raises describes
        # the config and arrives beside the issues its siblings reported. The
        # annotation is the whole reading here, since the factory raised ahead of
        # making the product, and it names the section a repaired factory builds.
        if _holds_sections(_UNSET, hint):
            collector.add_pending_lifecycle(path)
        return _reject(collector, path, f"default_factory raised {_exception_label(exc)}: {_describe(exc)}")
    if not validate_authored_value(produced, hint, path, collector, label=FACTORY_LABEL, depth=depth):
        # The product is put down before the lifecycle walk reaches it. What the
        # factory made answers for the sections already built, and the annotation
        # answers for the section a repaired factory goes on to build, which is the
        # reading a rejected leaf under a section annotation needs.
        if _holds_sections(produced, hint):
            collector.add_pending_lifecycle(path)
        return _UNSET
    _check_authored_sections(produced, path, collector)
    return produced


def _coerce(
    value: Any,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    *,
    data_chain: tuple[int, ...] = (),
    depth: int = 0,
) -> Any:
    """Convert one value toward its annotated type, recording issues on failure.

    Args:
      value (Any): The raw value from the config mapping.
      hint (Any): The resolved type hint the value must satisfy.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      data_chain (tuple[int, ...] = ()): Ids of the supplied mappings open on this
        branch, carried to any section this value reaches.
      depth (int = 0): How deep in the plain document this value sits.

    Returns:
      Any: The coerced value, or the ``_UNSET`` sentinel when coercion failed.
    """

    # A field can only carry an array annotation if its backend module is
    # loaded (the annotation object references it), so when no backend is present
    # there is nothing to match and the per-value inspection is skipped entirely.
    if collector.backend.active:
        matched_array = array_match(hint, collector.backend)
        if matched_array.matched:
            if matched_array.spec is None:
                return _reject(collector, path, typing.cast("str", matched_array.error))
            result = _arrays.coerce_array(value, matched_array.spec, path, collector.add)
            if result is _arrays.FAILED:
                return _UNSET
            # The array the load built writes one list level per axis, which the
            # budget pays for exactly as it pays for the same nesting written as
            # lists, so a value the marshal walk would decline is declined here.
            encoded = _arrays.encoded_array_depth(result)
            if encoded is not None and depth + encoded > MAX_PLAIN_DEPTH:
                return _reject(collector, path, plain_depth_message())
            return result

    # The structural dispatch (strip Annotated, origin/args, dataclass/container
    # detection) is a pure function of the hint, so it is computed once per hint
    # and reused across every value coerced against it. Branches are ordered by
    # frequency: leaves and nested dataclasses/containers dominate real configs.
    plan = _classify_hint(hint)
    kind = plan.kind

    if kind is _HintKind.SCALAR:
        return _coerce_scalar(value, plan.stripped, path, collector)
    if kind is _HintKind.DATACLASS:
        return _build(typing.cast("type[Any]", plan.dataclass_type), value, path, collector, (), data_chain, depth)
    if kind is _HintKind.CONTAINER:
        return _coerce_container(value, plan.stripped, plan.origin, plan.args, path, collector, data_chain, depth)
    if kind is _HintKind.UNION:
        return _coerce_union(value, plan.stripped, plan.args, path, collector, data_chain, depth)
    if kind is _HintKind.CONFIG_VALUE:
        return _coerce_config_value(value, plan.stripped, path, collector, depth=depth)
    if kind is _HintKind.NONE:
        if value is None:
            return None
        return _reject(collector, path, f"expected None, got {_typename(value)}")
    if kind is _HintKind.LITERAL:
        # Exact-type match keeps bool True distinct from int 1, which compare equal.
        if any(value == option and type(value) is type(option) for option in plan.args):
            return value
        return _reject(collector, path, f"expected one of {_hint_name(plan.stripped)}, got {value!r}")
    return _reject(collector, path, unsupported_hint_message(plan.stripped))


def _coerce_config_value(
    value: Any,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    *,
    depth: int = 0,
    seen: tuple[int, ...] = (),
) -> Any:
    """Accept a value that already carries a plain-data form.

    ``ConfigValue`` names the domain a config file carries: JSON scalars, lists
    of them, and string-keyed mappings of them. A sequence rebuilds as a list so
    the value round-trips to the same plain form it was written as, and every
    leaf is checked for the JSON representation the fingerprint depends on.
    ``ConfigScalar`` admits the leaf half of that domain alone.

    Nesting is bounded twice over: an identity stack ends a self-referential
    structure at the value that closes the loop, and a depth limit ends one that
    is merely deeper than the walk supports. Both report at the path they reach.

    Args:
      value (Any): The raw value from the config mapping.
      hint (Any): ``ConfigValue`` or ``ConfigScalar``, deciding whether
        containers are admitted.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      depth (int = 0): Nesting depth reached so far.
      seen (tuple[int, ...] = ()): Ids of the containers open on this branch.

    Returns:
      Any: The value in its plain form, or the ``_UNSET`` sentinel when it holds
        something outside the domain.
    """
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if math.isfinite(value):
            return value
        return _reject(collector, path, f"expected a finite float, got {value!r}")

    if hint is ConfigScalar:
        return _reject(collector, path, plain_scalar_message(value))
    if depth >= MAX_PLAIN_DEPTH:
        return _reject(collector, path, plain_depth_message())
    if id(value) in seen:
        return _reject(collector, path, plain_cycle_message())
    branch = (*seen, id(value))

    if isinstance(value, Mapping):
        mapping: dict[str, Any] = {}
        failed = False
        for key, item in value.items():
            if type(key) is not str:
                # The key is named by its type rather than its text, so a key
                # whose ``__str__`` raises is reported instead of run.
                collector.add(path, plain_key_message(key))
                failed = True
                continue
            item_path = _join(path, key)
            coerced = _coerce_config_value(item, hint, item_path, collector, depth=depth + 1, seen=branch)
            if coerced is _UNSET:
                failed = True
                continue
            mapping[key] = coerced
        return _UNSET if failed else mapping

    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        failed = False
        for index, item in enumerate(value):
            coerced = _coerce_config_value(item, hint, _join(path, str(index)), collector, depth=depth + 1, seen=branch)
            if coerced is _UNSET:
                failed = True
                continue
            items.append(coerced)
        return _UNSET if failed else items

    return _reject(collector, path, plain_data_message(value))


_NUMERIC_TYPES: tuple[type, ...] = (bool, int, float)
"""The classes a plain document tells apart on its own among the numbers.

A parser reads each of these as itself, so a value's own class names which of
them a file carried. Coercion converts between them where the conversion is
lossless, which is what makes the order they are tried in matter.
"""


def _numeric_first(value: Any, members: list[Any]) -> list[Any]:
    """Order union members so one naming the value's own numeric class goes first.

    A numeric member accepts a value of another numeric class by conversion, since
    an integral float lands on an ``int`` field so that ``1e6`` is accepted. A
    union naming two of them would therefore answer by declaration order alone and
    rebuild a value as the other member, which is what a round trip through a file
    would show. Trying the member that names the class the plain form already
    carries settles the pair in both declaration orders; every other member keeps
    the declared order.

    Two distinct classes from that family are what makes the reordering apply, so
    the count runs over the classes rather than over the members naming them: one
    numeric member beside a member of any other kind is a union whose order the
    author chose among kinds, and ``Number | int`` reads ``1`` as the declared
    ``Number`` exactly as it reads every other value by declaration order, as does
    the same union spelled with several ``Annotated`` variants of that one class.
    Where two of the classes are named, the member carrying the value's class is
    tried ahead of every other member, an earlier member of another kind included,
    since that is what settles the conversion between them.

    Args:
      value (Any): The raw value from the config mapping.
      members (list[Any]): The union's members, ``None`` already removed.

    Returns:
      list[Any]: The members to try, in the order to try them.
    """
    carried = type(value)
    if carried not in _NUMERIC_TYPES:
        return members
    stripped = [_strip_annotated(member) for member in members]
    if len({member for member in stripped if member in _NUMERIC_TYPES}) < 2:
        return members
    preferred = [member for member, bare in zip(members, stripped, strict=True) if bare is carried]
    if len(preferred) == 0:
        return members
    return [*preferred, *(member for member in members if member not in preferred)]


def _coerce_union(
    value: Any,
    hint: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
    data_chain: tuple[int, ...] = (),
    depth: int = 0,
) -> Any:
    """Coerce a value against a union, accepting the first member that fits cleanly.

    A failed union reports the member that came closest -- the one whose trial
    collected the fewest issues, declaration order breaking a tie -- so the report
    carries one branch's detail rather than every branch's, and the detail names
    the branch it came from.

    Args:
      value (Any): The raw value from the config mapping.
      hint (Any): The union type hint, used for the error message.
      args (tuple[Any, ...]): The union's member types.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      data_chain (tuple[int, ...] = ()): Ids of the supplied mappings open on this branch.
      depth (int = 0): How deep in the plain document this value sits.

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
        return _coerce(value, non_none[0], path, collector, data_chain=data_chain, depth=depth)
    attempts: list[tuple[Any, _IssueCollector]] = []
    for candidate in _numeric_first(value, non_none):
        # Probe each member with a throwaway collector so member-level failures
        # stay silent; the first clean conversion wins. Member order is precedence.
        # It inherits the operation's backend snapshot so array gating stays consistent.
        trial = _IssueCollector(collector.backend)
        result = _coerce(value, candidate, path, trial, data_chain=data_chain, depth=depth)
        if result is not _UNSET and trial.clean():
            return result
        attempts.append((candidate, trial))

    failures = [attempt for attempt in attempts if not attempt[1].clean()]
    if len(failures) == 0:
        return _reject(collector, path, f"expected {_hint_name(hint)}, got {_typename(value)}")
    # min is stable, so equal issue counts keep declaration order.
    candidate, best = min(failures, key=lambda attempt: len(attempt[1].issues))
    count = len(best.issues)
    noun = "issue" if count == 1 else "issues"
    collector.add(path, f"expected {_hint_name(hint)}; best match {_hint_name(candidate)} failed with {count} {noun}")
    # A member that converted cleanly returned above; this is the best of the
    # members that failed, and its diagnostics are the ones the report names, so
    # its pending paths travel with its issues. Every other member stays private,
    # pending paths included.
    collector.adopt(best)
    return _UNSET


def _coerce_items(
    items: list[Any],
    element_hints: list[Any],
    path: str,
    collector: _IssueCollector,
    data_chain: tuple[int, ...] = (),
    depth: int = 0,
) -> Any:
    """Coerce positional items against per-index hints, one dotted path each.

    Args:
      items (list[Any]): The raw elements to coerce.
      element_hints (list[Any]): The type hint for each element, aligned by index
        with ``items``.
      path (str): Dotted path of the container holding these items.
      collector (_IssueCollector): Destination for any issues found.
      data_chain (tuple[int, ...] = ()): Ids of the supplied mappings open on this branch.
      depth (int = 0): How deep in the plain document these items sit.

    Returns:
      Any: The list of coerced items, or the ``_UNSET`` sentinel when any element
        failed. Every element is visited so one pass reports all of them.
    """
    coerced: list[Any] = []
    failed = False
    for index, (item, element_hint) in enumerate(zip(items, element_hints, strict=True)):
        result = _coerce(item, element_hint, _join(path, str(index)), collector, data_chain=data_chain, depth=depth)
        if result is _UNSET:
            failed = True
            continue
        coerced.append(result)
    return _UNSET if failed else coerced


def _describe(exc: BaseException) -> str:
    """Render an exception for a message without letting it fail again.

    An exception raised by user code carries a ``__str__`` of its own, and that
    call, the ``len`` of what it returns, and the rendering of what it returns
    are each user code. The text is copied into an exact ``str`` so the last of
    those cannot fail in the caller, the class name answers when either earlier
    one does, and a fixed phrase answers when reading the class name raises too,
    so reporting one failure stays one failure. A ``MemoryError`` or
    ``SystemError`` from any of those calls travels on, as it does from building
    a container, and so does a ``BaseException`` outside ``Exception``.

    Args:
      exc (BaseException): The caught exception.

    Returns:
      str: The exception's text, or its class name when rendering it fails.
    """
    try:
        text = str(exc)
        if len(text) > 0:
            # str.__str__ copies the characters into an exact str, so a subclass
            # carrying rendering hooks of its own cannot fail again downstream.
            return str.__str__(text)
    except RESOURCE_ERRORS:
        raise
    except Exception:
        pass
    try:
        return str.__str__(type(exc).__name__)
    except RESOURCE_ERRORS:
        raise
    except Exception:
        return "an exception that could not be described"


def _coerce_container(
    value: Any,
    hint: Any,
    origin: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
    data_chain: tuple[int, ...] = (),
    depth: int = 0,
) -> Any:
    """Coerce a value into an annotated container, recursing into its elements.

    Args:
      value (Any): The raw value from the config mapping.
      hint (Any): The container type hint, used for the error message.
      origin (Any): The container's unsubscripted origin type.
      args (tuple[Any, ...]): The container's element type arguments.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      data_chain (tuple[int, ...] = ()): Ids of the supplied mappings open on this branch.
      depth (int = 0): How deep in the plain document this container sits.

    Returns:
      Any: The coerced container, or the ``_UNSET`` sentinel when coercion failed.
    """
    if depth >= MAX_PLAIN_DEPTH:
        return _reject(collector, path, plain_depth_message())
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
            coerced_key = _coerce(raw_key, key_hint, item_path, collector, data_chain=data_chain, depth=depth + 1)
            coerced_value = _coerce(raw_value, value_hint, item_path, collector, data_chain=data_chain, depth=depth + 1)
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
            # Fixed-length tuple: each position carries its own meaning, so an
            # input whose iteration order the caller did not express would fill
            # those positions arbitrarily. A set and a frozenset are exactly that,
            # and their order varies from run to run.
            if isinstance(value, (set, frozenset)):
                return _reject(
                    collector,
                    path,
                    f"expected an ordered sequence for {_hint_name(hint)}, got {_typename(value)}; "
                    f"each position of this tuple carries its own meaning, so write the items in a list",
                )
            if len(items) != len(args):
                return _reject(collector, path, f"expected {len(args)} items for {_hint_name(hint)}, got {len(items)}")
            element_hints = list(args)
    else:
        element_hint = args[0] if len(args) >= 1 else Any
        element_hints = [element_hint] * len(items)

    coerced_items = _coerce_items(items, element_hints, path, collector, data_chain, depth + 1)
    if coerced_items is _UNSET:
        return _UNSET
    builder = _SEQUENCE_BUILDERS.get(origin, list)
    try:
        return builder(coerced_items)
    except RESOURCE_ERRORS:
        # The interpreter is reporting on itself rather than on the config, so
        # this travels to the caller as the runtime failure it is.
        raise
    except Exception as exc:
        # Building a set hashes each element, and coercion hands back a value that
        # already satisfies its annotation, so an accepted subclass carrying its
        # own hash reaches this point. Whatever that hash does arrives as an issue
        # under the annotation as written.
        return _reject(collector, path, f"cannot build {_hint_name(hint)}: {_describe(exc)}")


def _coerce_scalar(value: Any, hint: Any, path: str, collector: _IssueCollector) -> Any:
    """Coerce a value toward a plain (unparameterized) annotated type.

    A short router over the supported leaf types: it dispatches to the enum,
    numeric, string, path, and ISO temporal branches in that fixed order (with
    ``datetime`` decided before ``date``, since ``datetime`` subclasses it).

    Args:
      value (Any): The raw value from the config mapping.
      hint (Any): The scalar type hint.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The coerced value, or the ``_UNSET`` sentinel when coercion failed.
    """
    # A numpy scalar can only exist when numpy is loaded; skip the check otherwise.
    if collector.backend.numpy is not None:
        is_numpy_scalar, normalized = _arrays.normalize_numpy_scalar(value)
        if is_numpy_scalar:
            # A supported numpy scalar feeds the ordinary rules as its exact Python
            # equivalent, so np.float32 lands on float fields and np.int64 on int.
            value = normalized

    if not isinstance(hint, type):
        return _reject(collector, path, unsupported_hint_message(hint))

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

    return _reject(collector, path, unsupported_hint_message(hint))


def _coerce_enum(value: Any, hint: type[Enum], path: str, collector: _IssueCollector) -> Any:
    """Resolve a value to an ``Enum`` member by value, then by member name.

    Resolving reads code the enum's author owns: a ``_missing_`` hook inside the
    value lookup, and the member mapping the name fallback and the option list read.
    The whole resolution runs under one containment, so what any of them raises
    describes the config and arrives as one issue at this value's path, beside the
    issues already collected.

    Args:
      value (Any): The raw value from the config mapping.
      hint (type[Enum]): The ``Enum`` subclass to resolve against.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The resolved enum member, or the ``_UNSET`` sentinel when no member
        matched.
    """
    try:
        return _resolve_enum_member(value, hint, path, collector)
    except RESOURCE_ERRORS:
        raise
    except Exception as exc:
        return _reject(
            collector,
            path,
            f"resolving enum {class_label(hint)} raised {_exception_label(exc)}: {_describe(exc)}; "
            f"leave the member values, the member names, and any _missing_ hook answering for the "
            f"values a file carries",
        )


def _resolve_enum_member(value: Any, hint: type[Enum], path: str, collector: _IssueCollector) -> Any:
    """Look a value up as a member value, then as a member name.

    Args:
      value (Any): The raw value from the config mapping.
      hint (type[Enum]): The ``Enum`` subclass to resolve against.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The resolved enum member, or the ``_UNSET`` sentinel when no member
        matched.
    """
    try:
        resolved = hint(value)
    except ValueError:
        resolved = None
    if type(resolved) is hint:
        # Lookup goes through the enum class, so what it hands back is confirmed
        # to belong to the annotation before it stands as the coerced value. The
        # check reads the result's own type rather than asking the class, since
        # the class's metaclass is the thing being confirmed.
        return resolved
    if isinstance(value, str) and value in hint.__members__:
        member = hint[value]
        if type(member) is hint:
            return member
    options = ", ".join(repr(member.value) for member in hint)
    return _reject(collector, path, f"expected one of {options} for enum {class_label(hint)}, got {value!r}")


def _coerce_numeric(value: Any, hint: type, path: str, collector: _IssueCollector) -> Any:
    """Coerce a value toward ``bool``, ``int``, or ``float``.

    ``bool`` is an ``int`` subclass, so it is kept off ``int`` / ``float`` fields
    explicitly, and whole-number floats land on ``int`` fields so forms like
    ``1e6`` are accepted.

    Args:
      value (Any): The raw value from the config mapping.
      hint (type): One of ``bool``, ``int``, or ``float``.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The coerced number, or the ``_UNSET`` sentinel when coercion failed.
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
      value (Any): The raw value from the config mapping.
      parser (Callable[[str], Any]): The ``fromisoformat`` parser for the target
        temporal type.
      label (str): The type name used in the issue message (``datetime`` /
        ``date`` / ``time``).
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      Any: The parsed temporal value, or the ``_UNSET`` sentinel when the value
        carried some other form.
    """
    if isinstance(value, str):
        try:
            return parser(value)
        except ValueError:
            return _reject(collector, path, f"expected an ISO 8601 {label} string, got {value!r}")
    return _reject(collector, path, f"expected an ISO 8601 {label} string, got {_typename(value)}")
