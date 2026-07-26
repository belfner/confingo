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
from confingo._errors import (
    _UNSET,
    ConfigError,
    _IssueCollector,
    _reject,
)
from confingo._schema import (
    _ClassifiedField,
    _classify_dataclass_fields,
    _is_dataclass_type,
    _join,
    _typename,
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


def _project_plain(value: Any, projection: _PlainProjection) -> Any:
    """Render a config value to plain data under one field projection.

    Args:
      value (Any): The config object or nested value to convert.
      projection (_PlainProjection): The field projection selecting which
        dataclass fields emit.

    Returns:
      Any: The converted plain-data structure.

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
      value (Any): The config object or nested value to convert.

    Returns:
      Any: The converted plain-data structure.

    Raises:
      ConfigError: When a value's type falls outside the supported set and has
        no plain-data form, or holds a non-finite float; the exception lists
        every issue found, each tagged with its dotted path.
    """
    return _project_plain(value, _PlainProjection.EXPORT)


def _to_plain(value: Any, path: str, collector: _IssueCollector, *, projection: _PlainProjection) -> Any:
    """Convert one value to plain data, recording issues with dotted paths.

    Args:
      value (Any): The config object or nested value to convert.
      path (str): Dotted path of this value, empty at the root.
      collector (_IssueCollector): Destination for any issues found.
      projection (_PlainProjection): The field projection carried through every
        recursive call, so a dataclass node emits only the fields the projection
        selects.

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
    # Reading the raw namespaces keeps a hostile metaclass __getattr__ dormant,
    # so a class object carried under Any is reported rather than left to raise.
    if not isinstance(value, type) and _is_dataclass_type(value_type):
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

    # Native arrays and numpy scalars only exist when a backend is loaded, so the
    # value-driven array handling is skipped entirely when none is present.
    if collector.backend.active:
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
      plain (Any): Plain JSON-safe data from a plain projection.

    Returns:
      str: Compact JSON with sorted mapping keys.
    """
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Any, *, length: int = 12) -> str:
    """Fingerprint a config with a stable digest over its canonical JSON form.

    The digest ranges over the hashing fields (``init=True``, ``compare=True``,
    effective hash enabled), so a ``compare=False`` or ``hash=False`` field is
    carried by ``to_dict`` yet excluded here. Mapping key order and set iteration
    order are normalized before hashing, so the digest is stable across processes.

    Args:
      config (Any): The config object to fingerprint.
      length (int = 12): Number of leading hex characters to return, by default 12.

    Returns:
      str: The truncated SHA-256 digest.
    """
    payload = _canonical_json(_project_plain(config, _PlainProjection.HASH))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
