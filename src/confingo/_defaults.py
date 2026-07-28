"""Strict validation of authored default values.

A supplied value travels through coercion; an authored default is kept as the
exact object the schema declares. It therefore has to arrive in the form the
engine would otherwise produce, which is what this module checks: the value
already carries the annotation's runtime representation, and it projects to
confingo's plain serializable form. Both gates run without converting anything,
so ``output_dir: Path = "runs"`` is an authoring error rather than a silent
promotion to ``Path("runs")``.

Direct ``field(default=...)`` values are checked during data-independent schema
preflight, where the value already exists. That check stays clear of the
construction engine: coercion, enum lookup hooks, field factories,
``__post_init__``, and ``__validate__`` belong to building a config and stay
unentered. Projecting the value to its plain form reads it through its own
accessors, so a mapping's ``items``, a property, and ``__str__`` answer for the
values they describe. A ``default_factory`` is checked in ``_core`` at the one
build that selects it, since calling a factory during preflight would run user
code on a cached path and duplicate its side effects.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    get_origin,
)

from confingo import _arrays
from confingo._errors import _IssueCollector
from confingo._schema import (
    MAX_PLAIN_DEPTH,
    _classify_dataclass_fields,
    _classify_hint,
    _hint_name,
    _HintKind,
    _join,
    _resolved_hints,
    _typename,
    plain_cycle_message,
    plain_data_message,
    plain_depth_message,
    plain_key_message,
    plain_scalar_message,
)
from confingo._serialize import (
    _PlainProjection,
    _to_plain,
)
from confingo.typing import ConfigScalar


DEFAULT_LABEL = "invalid authored default"
"""Message prefix for a value written as ``field(default=...)`` or ``x: T = value``."""

FACTORY_LABEL = "invalid default_factory value"
"""Message prefix for the one value a selected ``default_factory`` produced."""

_EXACT_SCALARS: dict[Any, type] = {bool: bool, int: int, float: float, str: str}
"""Builtin scalar annotations mapped to the exact type a default must already be.

Membership is by exact type on the value side, which keeps ``True`` off an
``int`` field and an ``int`` off a ``float`` field: coercion promotes both, and a
default is never coerced.
"""

_SEQUENCE_TYPES: dict[Any, type] = {tuple: tuple, set: set, frozenset: frozenset}
"""Sequence origins mapped to the exact type a default must already be.

Every other sequence origin, ``list`` and ``Sequence`` among them, builds a
``list``.
"""


def validate_authored_value(
    value: Any,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    *,
    label: str,
    depth: int = 0,
) -> bool:
    """Validate one authored value against its annotation and its plain form.

    Args:
      value (Any): The authored object, checked in place.
      hint (Any): The resolved type hint the value is declared under.
      path (str): Dotted path of the field carrying the value.
      collector (_IssueCollector): Destination for any issues found, each message
        carried under ``label``.
      label (str): Prefix naming where the value came from, ``DEFAULT_LABEL`` or
        ``FACTORY_LABEL``.
      depth (int = 0): How deep in the plain document this value sits, so the
        nesting budget is the one the whole-config walks spend.

    Returns:
      bool: True when the value passed both gates, False when issues were added.
    """
    trial = _IssueCollector(collector.backend)
    _check_runtime_form(value, hint, path, trial, set(), depth)
    if trial.clean():
        # The same walk to_dict runs, at the depth to_dict reaches this value at, so
        # a default that validates is a default the config can be written back out
        # with.
        _to_plain(value, path, trial, projection=_PlainProjection.EXPORT, depth=depth)
    for issue in trial.issues:
        collector.add(issue.path, f"{label}: {issue.message}")
    return trial.clean()


def _mismatch(hint: Any, value: Any) -> str:
    """Build the message for a value that would need coercion to fit its annotation.

    Args:
      hint (Any): The resolved type hint the value is declared under.
      value (Any): The authored object.

    Returns:
      str: The message naming the annotation, the value's type, and the rule.
    """
    return (
        f"expected a value already matching {_hint_name(hint)}, got {_typename(value)}; "
        f"defaults are validated as written"
    )


def _check_runtime_form(
    value: Any, hint: Any, path: str, collector: _IssueCollector, seen: set[int], depth: int = 0
) -> None:
    """Check that a value already carries the runtime representation ``hint`` names.

    Args:
      value (Any): The authored object.
      hint (Any): The resolved type hint.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
      seen (set[int]): Ids of the config objects on the current branch, so a
        self-referential default terminates.
      depth (int = 0): How deep in the plain document this value sits.
    """
    if collector.backend.active:
        match = _arrays.inspect_annotation(hint)
        if match.matched:
            _check_array(value, match, hint, path, collector)
            return

    plan = _classify_hint(hint)
    kind = plan.kind
    if kind is _HintKind.CONFIG_VALUE:
        _check_plain_domain(value, plan.stripped, path, collector, depth=depth)
        return
    if kind is _HintKind.NONE:
        if value is not None:
            collector.add(path, _mismatch(hint, value))
        return
    if kind is _HintKind.LITERAL:
        # Exact-type match keeps bool True distinct from int 1, as coercion does.
        if not any(value == option and type(value) is type(option) for option in plan.args):
            collector.add(path, f"expected one of {_hint_name(plan.stripped)}, got {value!r}")
        return
    if kind is _HintKind.UNION:
        _check_union(value, plan.stripped, plan.args, path, collector, seen, depth)
        return
    if kind is _HintKind.DATACLASS:
        _check_section(value, plan.dataclass_type, hint, path, collector, seen, depth)
        return
    if kind is _HintKind.CONTAINER:
        _check_container(value, plan.stripped, plan.origin, plan.args, path, collector, seen, depth)
        return
    _check_scalar(value, plan.stripped, path, collector)


def _check_plain_domain(
    value: Any,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    *,
    depth: int = 0,
    seen: tuple[int, ...] = (),
) -> None:
    """Check that an authored value already carries the exact plain form its alias names.

    ``ConfigValue`` and ``ConfigScalar`` name a domain rather than one type, so
    the check walks the value and requires each part to be a member as written.
    Coercion admits a tuple by rebuilding it as a list, and admits a ``Path`` or a
    temporal value by writing it as text; a default is never coerced, so each of
    those is reported here with the plain form to write instead.

    Args:
      value (Any): The authored object.
      hint (Any): ``ConfigValue`` or ``ConfigScalar``, deciding whether containers
        are admitted.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
      depth (int = 0): Nesting depth reached so far.
      seen (tuple[int, ...] = ()): Ids of the containers open on this branch.
    """
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return
    if value_type is float:
        if not math.isfinite(value):
            collector.add(path, f"expected a finite float, got {value!r}")
        return
    if hint is ConfigScalar:
        collector.add(path, plain_scalar_message(value))
        return
    if depth >= MAX_PLAIN_DEPTH:
        collector.add(path, plain_depth_message())
        return
    if id(value) in seen:
        collector.add(path, plain_cycle_message())
        return
    branch = (*seen, id(value))
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                collector.add(path, plain_key_message(key))
                continue
            _check_plain_domain(item, hint, _join(path, key), collector, depth=depth + 1, seen=branch)
        return
    if value_type is list:
        for index, item in enumerate(value):
            _check_plain_domain(item, hint, _join(path, str(index)), collector, depth=depth + 1, seen=branch)
        return
    collector.add(path, plain_data_message(value))


def _check_array(value: Any, match: _arrays.AnnotationMatch, hint: Any, path: str, collector: _IssueCollector) -> None:
    """Check an array or tensor default against its dtype, shape, and finiteness.

    The classified annotation is applied through the same coercion the engine
    runs, which returns the value itself when it already satisfies the
    annotation. Any other successful result means a conversion happened, so the
    default was written in a form the engine would have had to change.

    Args:
      value (Any): The authored array or tensor.
      match (_arrays.AnnotationMatch): The classified array annotation.
      hint (Any): The resolved type hint, used in the mismatch message.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
    """
    if match.spec is None:
        collector.add(path, str(match.error))
        return
    trial = _IssueCollector(collector.backend)
    result = _arrays.coerce_array(value, match.spec, path, trial.add)
    if result is _arrays.FAILED:
        collector.extend(trial.issues)
        return
    if result is not value:
        collector.add(path, _mismatch(hint, value))


def _check_union(
    value: Any,
    hint: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
    seen: set[int],
    depth: int = 0,
) -> None:
    """Check a value against a union, accepting the first member it already fits.

    Args:
      value (Any): The authored object.
      hint (Any): The union type hint, used in the mismatch message.
      args (tuple[Any, ...]): The union's member types.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
      seen (set[int]): Ids of the config objects on the current branch.
      depth (int = 0): How deep in the plain document this value sits.
    """
    for member in args:
        trial = _IssueCollector(collector.backend)
        _check_runtime_form(value, member, path, trial, seen, depth)
        if trial.clean():
            return
    collector.add(path, _mismatch(hint, value))


def _check_section(
    value: Any,
    section_cls: type[Any] | None,
    hint: Any,
    path: str,
    collector: _IssueCollector,
    seen: set[int],
    depth: int = 0,
) -> None:
    """Check a nested config object default, recursing into its constructor fields.

    The annotated class is required exactly, because that is the class the engine
    builds and the class ``to_dict`` projects through. A subclass instance carries
    fields the annotation does not know, which export and then fail to reload as
    unknown keys. A union naming the subclass is the way to allow one.

    Args:
      value (Any): The authored section object.
      section_cls (type[Any] | None): The dataclass the annotation names.
      hint (Any): The resolved type hint, used in the mismatch message.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
      seen (set[int]): Ids of the config objects on the current branch, so a
        section holding itself terminates.
      depth (int = 0): How deep in the plain document this section sits.
    """
    if section_cls is None or type(value) is not section_cls:
        collector.add(path, _mismatch(hint, value))
        return
    if id(value) in seen:
        return
    branch = seen | {id(value)}
    hints = _resolved_hints(section_cls)
    for classified in _classify_dataclass_fields(section_cls).init_fields:
        name = classified.definition.name
        held = getattr(value, name, None)
        _check_runtime_form(held, hints[name], _join(path, name), collector, branch, depth + 1)


def _check_container(
    value: Any,
    hint: Any,
    origin: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
    seen: set[int],
    depth: int = 0,
) -> None:
    """Check a container default's own type, its arity, and each element it holds.

    The accepted type is the one the engine builds for the annotation, so a list
    default satisfies ``list[T]`` and ``Sequence[T]`` while a tuple default does
    not.

    Args:
      value (Any): The authored container.
      hint (Any): The container type hint, used in the mismatch message.
      origin (Any): The container's unsubscripted origin type.
      args (tuple[Any, ...]): The container's element type arguments.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
      seen (set[int]): Ids of the config objects on the current branch.
      depth (int = 0): How deep in the plain document this container sits.
    """
    if origin in (dict, Mapping):
        # Construction builds a dict from whatever mapping the input carried, so
        # dict is the runtime form both a Mapping annotation and a dict one hold.
        if type(value) is not dict:
            collector.add(path, _mismatch(hint, value))
            return
        value_hint = args[1] if len(args) == 2 else Any
        for key, item in value.items():
            item_path = _join(path, str(key))
            if type(key) is not str:
                collector.add(item_path, f"expected a str mapping key, got {_typename(key)}")
                continue
            _check_runtime_form(item, value_hint, item_path, collector, seen, depth + 1)
        return

    built_type = _SEQUENCE_TYPES.get(origin, list)
    if type(value) is not built_type:
        collector.add(path, _mismatch(hint, value))
        return

    if origin is tuple:
        element_hints = _tuple_element_hints(value, hint, args, path, collector)
        if element_hints is None:
            return
    else:
        element_hint = args[0] if len(args) >= 1 else Any
        element_hints = [element_hint] * len(value)

    if isinstance(value, (set, frozenset)):
        # Set iteration order is unstable, so an element index would name a
        # different element on each run; elements carry the set's own path.
        # Whether the elements survive a load is settled by the annotation at
        # schema preflight, so each element is judged on its runtime form alone.
        element_hint = args[0] if len(args) >= 1 else Any
        for item in value:
            _check_runtime_form(item, element_hint, path, collector, seen, depth + 1)
        return
    for index, (item, element_hint) in enumerate(zip(value, element_hints, strict=True)):
        _check_runtime_form(item, element_hint, _join(path, str(index)), collector, seen, depth + 1)


def _tuple_element_hints(
    value: tuple[Any, ...],
    hint: Any,
    args: tuple[Any, ...],
    path: str,
    collector: _IssueCollector,
) -> list[Any] | None:
    """Align a tuple default's items with per-position hints, checking arity.

    Args:
      value (tuple[Any, ...]): The authored tuple.
      hint (Any): The tuple type hint, used in the arity message.
      args (tuple[Any, ...]): The tuple's element type arguments.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.

    Returns:
      list[Any] | None: One hint per item, or None when the arity is wrong.
    """
    if len(args) == 2 and args[1] is Ellipsis:
        return [args[0]] * len(value)
    if len(args) == 0:
        if get_origin(hint) is not tuple:
            # Bare tuple: each item passes through under Any.
            return [Any] * len(value)
        if len(value) != 0:
            collector.add(path, f"expected 0 items for {_hint_name(hint)}, got {len(value)}")
            return None
        return []
    if len(value) != len(args):
        collector.add(path, f"expected {len(args)} items for {_hint_name(hint)}, got {len(value)}")
        return None
    return list(args)


def _check_scalar(value: Any, hint: Any, path: str, collector: _IssueCollector) -> None:
    """Check a leaf default against the exact runtime type its annotation names.

    Args:
      value (Any): The authored leaf value.
      hint (Any): The scalar type hint.
      path (str): Dotted path of the value.
      collector (_IssueCollector): Destination for any issues found.
    """
    if not isinstance(hint, type):
        # An unsupported annotation is already reported by schema validation.
        return
    exact = _EXACT_SCALARS.get(hint)
    if exact is not None:
        if type(value) is not exact:
            collector.add(path, _mismatch(hint, value))
        elif exact is float and not math.isfinite(value):
            collector.add(path, f"expected a finite float, got {value!r}")
        return
    if issubclass(hint, Enum) or issubclass(hint, Path) or issubclass(hint, (dt.datetime, dt.time)):
        if not isinstance(value, hint):
            collector.add(path, _mismatch(hint, value))
        return
    if issubclass(hint, dt.date):
        # datetime subclasses date, and coercion keeps it off a plain date field,
        # so a datetime default is a mismatch rather than a silent truncation.
        if not isinstance(value, hint) or isinstance(value, dt.datetime):
            collector.add(path, _mismatch(hint, value))
        return
    if not isinstance(value, hint):
        collector.add(path, _mismatch(hint, value))
