"""Serialization of config objects to plain data, and the stable fingerprint.

``to_dict`` renders a built config into plain JSON-safe Python data in
field-declaration order; ``config_hash`` fingerprints a config over the same
plain form's canonical JSON. Both run one recursive walk (``_to_plain``) under a
field projection selecting which dataclass fields emit, so export, the equality
serialized fallback, and the fingerprint agree on how each value serializes.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from confingo import _arrays
from confingo._choice import (
    group_record,
    is_group,
    variant_tag,
)
from confingo._errors import (
    _UNSET,
    ConfigError,
    _IssueCollector,
    _reject,
    class_label,
)
from confingo._schema import (
    MAX_PLAIN_DEPTH,
    MAX_RENDER_HOPS,
    _ClassifiedField,
    _classify_dataclass_fields,
    _is_dataclass_type,
    _join,
    _typename,
    plain_cycle_message,
    plain_depth_message,
    render_hop_message,
)


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


def _projected_fields(config_cls: type[Any], projection: _PlainProjection) -> tuple[_ClassifiedField, ...]:
    """Select the classified fields a serialization projection emits.

    Args:
      config_cls (type[Any]): The dataclass whose fields to select.
      projection (_PlainProjection): The serialization projection.

    Returns:
      tuple[_ClassifiedField, ...]: The classified fields for the projection, in
        declaration order.
    """
    classification = _classify_dataclass_fields(config_cls)
    if projection is _PlainProjection.EXPORT:
        return classification.init_fields
    if projection is _PlainProjection.COMPARE:
        return classification.compared
    return classification.hashed


def _project_plain(value: Any, projection: _PlainProjection, depth: int = 0) -> Any:
    """Render a config value to plain data under one field projection.

    Args:
      value (Any): The config object or nested value to convert.
      projection (_PlainProjection): The field projection selecting which
        dataclass fields emit.
      depth (int = 0): How deep in the plain document this value sits, so a walk
        resumed partway spends the budget its caller already spent.

    Returns:
      Any: The converted plain-data structure.

    Raises:
      ConfigError: When a value's type falls outside the supported set and has
        no plain-data form, or holds a non-finite float; the exception lists
        every issue found, each tagged with its dotted path.
    """
    collector = _IssueCollector()
    result = _to_plain(value, "", collector, projection=projection, depth=depth)
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
    for every field whose annotation names a supported type, since ``from_dict``
    rebuilds each container from its annotation. A ``ConfigValue`` field returns
    in the plain form it was written as, so a tuple supplied to one returns as a
    list; annotate such a field with a container type to restore its exact type.

    The invariant reads over a value already carrying the type its annotation
    names, which is what a load builds and what a validated authored default
    carries. This renders the value it is given: Python's numeric tower lets a
    type checker accept ``x: float = 1``, and the ``int`` that field then holds
    is written as ``1`` and reloads as ``1.0``.

    Args:
      value (Any): The config object or nested value to convert.

    Returns:
      Any: The converted plain-data structure.

    Raises:
      ConfigError: When a value's type falls outside the supported set and has
        no plain-data form, or holds a non-finite float; the exception lists
        every issue found, each tagged with its dotted path.
    """
    return _project_plain(value, _PlainProjection.EXPORT)


def _array_plain_form(
    value: Any,
    path: str,
    collector: _IssueCollector,
    *,
    projection: _PlainProjection,
    depth: int,
    seen: tuple[int, ...],
    renders: int,
) -> Any:
    """Render a native array to its plain form, bounded and checked.

    An array writes one list level per axis up to the first empty one, which the
    nesting budget pays for exactly as it pays for the same nesting written as
    lists, and the shape answers it before anything is materialized. ``tolist`` belongs to the value's
    own class, so what a subclass produces is walked the way any other supplied
    value is; an exact backend class writes the form its backend documents and is
    returned as it stands.

    Args:
      value (Any): The value being marshalled.
      path (str): Dotted path of this value.
      collector (_IssueCollector): Destination for any issues found.
      projection (_PlainProjection): The field projection carried through.
      depth (int): How deep in the plain document this value sits.
      seen (tuple[int, ...]): Ids of the containers open on this branch.
      renders (int): Array-into-array render hops left before the walk reports.

    Returns:
      Any: ``_arrays.NOT_ARRAY`` when the value is unrelated to the loaded
        backends, the plain form when it renders, or the ``_UNSET`` sentinel.
    """
    encoded = _arrays.encoded_array_depth(value)
    if encoded is not None and depth + encoded > MAX_PLAIN_DEPTH:
        return _reject(collector, path, plain_depth_message())
    rendered = _arrays.array_to_plain(value, path, collector.add)
    if rendered is _arrays.NOT_ARRAY:
        return _arrays.NOT_ARRAY
    if rendered is _arrays.FAILED:
        return _UNSET
    if not _arrays.writes_its_own_plain_form(value):
        return rendered
    if not _arrays.is_array_value(rendered):
        # The class reached lists and numbers, so this render followed into no
        # further array and the walk carries on with the hops it still holds.
        return _to_plain(rendered, path, collector, projection=projection, depth=depth, seen=seen, renders=renders)
    if renders <= 0:
        # The class answered with another array, which answered with another, so
        # the chain writes no plain form for the walk to reach.
        return _reject(collector, path, render_hop_message())
    return _to_plain(rendered, path, collector, projection=projection, depth=depth, seen=seen, renders=renders - 1)


def _opened_node(value_type: type[Any]) -> dict[str, Any] | None:
    """Open the mapping one config section renders into.

    A variant's selection string comes from the object's own class rather than
    from the annotation it was reached through, so a variant rendered on its own
    carries the string that reads it back. It leads the mapping, which is where a
    file states which section the rest of the keys describe.

    Args:
      value_type (type[Any]): The class of the section being rendered.

    Returns:
      dict[str, Any] | None: The mapping to fill, carrying the selection when the
        class is a variant, or None when the class is a group base and names no
        variant for a section to carry.
    """
    selection = variant_tag(value_type)
    if selection is None:
        return None if is_group(value_type) else {}
    group, tag = selection
    record = group_record(group)
    return {} if record is None else {record.tag_key: tag}


def _group_instance_message(group: type[Any]) -> str:
    """Build the rejection for a variant-group base rendered as a config section.

    A group base is an ordinary dataclass, so constructing one is valid Python
    that a type checker accepts. It names no variant, so the section it would
    render carries no selection and the next load rejects what this one wrote.

    Args:
      group (type[Any]): The group base the value is an instance of.

    Returns:
      str: The rejection naming the group and the variants to build instead.
    """
    record = group_record(group)
    variants = "no registered variants"
    if record is not None and len(record.by_tag) > 0:
        variants = ", ".join(class_label(record.by_tag[tag]) for tag in sorted(record.by_tag))
    return (
        f"{class_label(group)} is a variant group standing for the sections behind it, and a config section "
        f"names one of them; build {variants} so the section carries the selection a load reads it back by"
    )


def _to_plain(
    value: Any,
    path: str,
    collector: _IssueCollector,
    *,
    projection: _PlainProjection,
    depth: int = 0,
    seen: tuple[int, ...] = (),
    renders: int = MAX_RENDER_HOPS,
) -> Any:
    """Convert one value to plain data, recording issues with dotted paths.

    Nesting is bounded twice over, the same way every other walk over a value is:
    an identity stack ends a structure that reaches itself at the value that
    closes the loop, and ``MAX_PLAIN_DEPTH`` ends one that is merely deeper than
    the walk supports. Both arrive as an issue at the path they reach.

    Args:
      value (Any): The config object or nested value to convert.
      path (str): Dotted path of this value, empty at the root.
      collector (_IssueCollector): Destination for any issues found.
      projection (_PlainProjection): The field projection carried through every
        recursive call, so a dataclass node emits only the fields the projection
        selects.
      depth (int = 0): Nesting depth reached so far.
      seen (tuple[int, ...] = ()): Ids of the containers open on this branch.
      renders (int = MAX_RENDER_HOPS): Array-into-array render hops left before
        the walk reports.

    Returns:
      Any: The converted plain-data structure, or the ``_UNSET`` sentinel when
        this value failed to serialize.
    """
    # Exact-builtin leaves are the majority of nodes and cannot also be a
    # dataclass, Enum, Path, temporal, array, numpy scalar, or mapping, so they
    # short-circuit the ordered isinstance chain below. Subclasses fall through
    # to the richer branches to preserve their behavior.
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if not math.isfinite(value):
            return _reject(collector, path, f"cannot serialize non-finite float {value!r}")
        return value
    if depth >= MAX_PLAIN_DEPTH:
        return _reject(collector, path, plain_depth_message())
    if id(value) in seen:
        return _reject(collector, path, plain_cycle_message())
    branch = (*seen, id(value))

    # Reading the raw namespaces keeps a hostile metaclass __getattr__ dormant, so
    # a class object carried in an open-data field is reported rather than left to
    # raise.
    if not isinstance(value, type) and _is_dataclass_type(value_type):
        node = _opened_node(value_type)
        if node is None:
            return _reject(collector, path, _group_instance_message(value_type))
        node_failed = False
        for classified in _projected_fields(type(value), projection):
            field_name = classified.definition.name
            item = _to_plain(
                getattr(value, field_name),
                _join(path, field_name),
                collector,
                projection=projection,
                depth=depth + 1,
                seen=branch,
                renders=renders,
            )
            if item is _UNSET:
                node_failed = True
                continue
            node[field_name] = item
        return _UNSET if node_failed else node
    if isinstance(value, Enum):
        return _to_plain(
            value.value, path, collector, projection=projection, depth=depth + 1, seen=branch, renders=renders
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.time)):
        # Covers date, datetime (itself a date subclass), and time.
        return value.isoformat()

    # Native arrays and numpy scalars only exist when a backend is loaded, so the
    # value-driven array handling is skipped entirely when none is present.
    if collector.backend.active:
        array_result = _array_plain_form(
            value, path, collector, projection=projection, depth=depth, seen=branch, renders=renders
        )
        if array_result is not _arrays.NOT_ARRAY:
            return array_result
        is_numpy_scalar, normalized = _arrays.normalize_numpy_scalar(value)
        if is_numpy_scalar:
            value = normalized

    if isinstance(value, Mapping):
        # Convert keys through the same rules; JSON carries string keys natively.
        mapping: dict[Any, Any] = {}
        mapping_failed = False
        for key, item in value.items():
            item_path = _join(path, str(key))
            plain_key = _to_plain(
                key, item_path, collector, projection=projection, depth=depth + 1, seen=branch, renders=renders
            )
            plain_item = _to_plain(
                item, item_path, collector, projection=projection, depth=depth + 1, seen=branch, renders=renders
            )
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
        elements = [
            _to_plain(item, path, collector, projection=projection, depth=depth + 1, seen=branch, renders=renders)
            for item in value
        ]
        if any(element is _UNSET for element in elements):
            return _UNSET
        # Sort by each element's canonical JSON text so the order is total and
        # stable across processes even for mixed-type sets whose elements are not
        # mutually orderable; equal sets must hash equal regardless of PYTHONHASHSEED.
        return sorted(elements, key=_canonical_json)
    if isinstance(value, (list, tuple)):
        items = [
            _to_plain(
                item,
                _join(path, str(index)),
                collector,
                projection=projection,
                depth=depth + 1,
                seen=branch,
                renders=renders,
            )
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

    This single encoding backs ``config_hash``, the equality serialized fallback,
    and the set-element ordering, so all three agree token for token: JSON keeps
    ``true`` / ``1`` / ``1.0`` distinct where Python ``==`` would conflate them,
    and equal configs therefore always fingerprint equally.

    The encoder is given no fallback converter, so it encodes what a plain
    projection produces and nothing else. Every value reaching it has already
    passed ``_to_plain``, which reports anything outside the plain-data domain, so
    a converter here could only turn a gap in that walk into a digest over text
    nothing can be rebuilt from.

    Args:
      plain (Any): Plain JSON-safe data from a plain projection.

    Returns:
      str: Compact JSON with sorted mapping keys.
    """
    return json.dumps(plain, sort_keys=True, separators=(",", ":"))


def config_hash(config: Any, *, length: int = 12) -> str:
    """Fingerprint a config with a stable digest over its canonical JSON form.

    The digest ranges over the hashing fields (``init=True``, ``compare=True``,
    effective hash enabled), so a ``compare=False`` or ``hash=False`` field is
    carried by ``to_dict`` yet excluded here. Mapping key order and set iteration
    order are normalized before hashing, so the digest is stable across processes.

    This is the value-identity operation, so reaching a config class here installs
    canonical equality on it and withdraws its hashing, the same ownership contract
    every other schema-processing route installs. That is what makes equal configs
    fingerprint equally: canonical equality and this digest read one plain form,
    where a generated ``__eq__`` reads Python ``==`` and holds values equal that
    the digest keeps apart.

    Args:
      config (Any): The config object to fingerprint.
      length (int = 12): Number of leading hex characters to return, from 1 to 64.

    Returns:
      str: The truncated SHA-256 digest.

    Raises:
      ConfigError: When ``length`` is outside 1 to 64, or is not an ``int``, or
        the class violates confingo's ownership of equality and hashing. The
        digest names a run directory, so a length other than the one asked for
        would name a different directory than the caller intended.
    """
    if type(length) is not int or not 1 <= length <= _DIGEST_LENGTH:
        raise ConfigError.single(
            f"config_hash length must be an int from 1 to {_DIGEST_LENGTH}, got {length!r}; "
            f"pass a length in that range, or leave it out for the 12-character default",
            context="config",
        )
    if not isinstance(config, type) and _is_dataclass_type(type(config)):
        from confingo._equality import _install_canonical_eq  # noqa: PLC0415

        _install_canonical_eq(type(config))
    payload = _canonical_json(_project_plain(config, _PlainProjection.HASH))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


_DIGEST_LENGTH = 64
"""Hex characters in the full SHA-256 digest, which bounds a requested length."""
