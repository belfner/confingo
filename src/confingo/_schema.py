"""Schema introspection over config dataclasses: hints, field metadata, validation.

This module reads the schema surface the marshal / unmarshal engine walks:
resolving a dataclass's annotations to runtime types (installing canonical
equality on first touch), classifying each field for the loading / export /
equality / fingerprint projections, and validating annotations against the
supported type set independently of any config data. Construction
(``_core``) and serialization (``_serialize``) consume these results.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import os
import re
import sys
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
)
from enum import (
    Enum,
    EnumType,
)
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
from confingo._choice import (
    group_record,
    is_group,
    registry_generation,
    variant_tag,
)
from confingo._errors import (
    ConfigError,
    ConfigIssue,
    _IssueCollector,
    class_label,
)
from confingo.typing import (
    ConfigScalar,
    ConfigValue,
)


if typing.TYPE_CHECKING:
    from confingo._backend import BackendSnapshot


_HINT_SLOT = "_confingo_resolved_hints"
"""Attribute a class carries its own resolved annotations under."""

_SCHEMA_SLOT = "_confingo_schema_issues"
"""Attribute a class carries its own schema-validation issues under."""

_FIELD_SLOT = "_confingo_classified_fields"
"""Attribute a class carries its own field classification under.

Each of the three per-class results is stored on the class it describes rather
than in a mapping this module roots, so its lifetime is the class's own: a class
defined in a notebook cell, a schema factory, or a plugin registry takes its
cached work with it when it is collected, a self-referential schema included.
Reads go through the class's own namespace rather than attribute lookup, so a
subclass computes its own results instead of inheriting a base's, and they read it
through ``type.__getattribute__`` so a metaclass ``__getattribute__`` stays out of
the read. The per-hint plan cache
alongside is keyed by identity rather than by a class and states its own bound.
"""


@dataclass(frozen=True)
class _CachedResult:
    """One per-class result, tagged so only confingo's own writes are read back.

    The three slot names are ordinary attribute names, and a schema class is free
    to bind any of them for its own purposes: as a field with a default, as a
    class attribute, or inherited from a base. Wrapping the result marks the
    binding as confingo's, and recording the class it describes keeps a value
    written for one class from being read on another that happens to expose it.

    Attributes:
      owner (type): The class this result describes.
      value (Any): The computed result.
      generation (int | None): The variant-registration generation the result was
        computed at, for a result that reads the registry; None for one that does
        not. A stored generation older than the current one marks the result
        stale, which is what keeps a whole-tree schema result honest while
        variants register as their modules import.
    """

    owner: type
    value: Any
    generation: int | None = None


def forget_schema_issues(config_cls: type[Any]) -> None:
    """Drop a class's cached schema issues, so the next check recomputes them.

    A binding the class made for itself is left in place, since it was never a
    cached result to begin with.

    Args:
      config_cls (type[Any]): The class whose cached issues to drop.
    """
    own = type.__getattribute__(config_cls, "__dict__")
    if not isinstance(own.get(_SCHEMA_SLOT), _CachedResult):
        return
    with contextlib.suppress(AttributeError, TypeError):
        delattr(config_cls, _SCHEMA_SLOT)


def _cached_on(config_cls: type[Any], slot: str, *, generation: int | None = None) -> Any:
    """Read a class's own cached result, ignoring anything a base carries.

    Args:
      config_cls (type[Any]): The class whose result to read.
      slot (str): The attribute the result is stored under.
      generation (int | None = None): The registration generation the caller
        requires, for a result that reads the variant registry. A result stored
        at any other generation is stale and reads as absent.

    Returns:
      Any: The cached result, or None when this class has none of its own, when
        the name holds something the class bound for itself, and when the stored
        result predates the registry the caller is asking about.
    """
    entry = type.__getattribute__(config_cls, "__dict__").get(slot)
    if isinstance(entry, _CachedResult) and entry.owner is config_cls:
        if generation is not None and entry.generation != generation:
            return None
        return entry.value
    return None


def _store_on(config_cls: type[Any], slot: str, result: Any, *, generation: int | None = None) -> None:
    """Store a class's own cached result on the class itself.

    The store is a cache rather than a contract, so it yields to the class in
    every case it cannot own the name: a class that refuses the attribute, and a
    class that binds the name for itself anywhere along its MRO, both keep working
    and recompute each time. A wrapper a base carries is confingo's own, so a
    subclass still writes the wrapper describing itself over it.

    Args:
      config_cls (type[Any]): The class the result describes.
      slot (str): The attribute to store it under.
      result (Any): The computed result.
      generation (int | None = None): The registration generation the result was
        computed at, for a result that reads the variant registry.
    """
    if _binds_its_own(config_cls, slot):
        return
    with contextlib.suppress(TypeError, AttributeError):
        setattr(config_cls, slot, _CachedResult(config_cls, result, generation))


def _binds_its_own(config_cls: type[Any], slot: str) -> bool:
    """Report whether a class or any base binds a slot name for its own purposes.

    Membership is what answers, so a binding whose value is ``None`` counts as the
    class's own. The whole MRO is read because an inherited binding is reachable
    on this class, and a slot descriptor a base declares is answered by that base
    rather than by anything written here.

    Args:
      config_cls (type[Any]): The class to inspect.
      slot (str): The attribute name to look for.

    Returns:
      bool: True when the name is bound to something other than a cached result.
    """
    return any(
        slot in namespace and not isinstance(namespace[slot], _CachedResult)
        for namespace in _class_namespaces(config_cls)
    )


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
    cached = _cached_on(config_cls, _HINT_SLOT)
    if cached is not None:
        return cached
    declared = _type_parameter_owner(config_cls)
    if declared is not None:
        # Checked before resolution: a postponed annotation naming a parameter
        # fails to resolve at all, and the parameter is what to report either way.
        owner, parameters, metaclass_of = declared
        message = type_parameter_message(owner, parameters, schema=metaclass_of)
        raise ConfigError.single(message, context="config schema")
    try:
        hints = get_type_hints(config_cls, include_extras=True)
    except NameError as exc:
        message = (
            f"cannot resolve the annotations of {class_label(config_cls)}: {exc}. "
            f"Declare config dataclasses at module level so their annotations resolve "
            f"in the defining module's namespace."
        )
        raise ConfigError.single(message, context="config schema") from exc
    from confingo._equality import _install_canonical_eq  # noqa: PLC0415

    _install_canonical_eq(config_cls)
    _store_on(config_cls, _HINT_SLOT, hints)
    return hints


_PARAMETER_ATTRIBUTES = ("__type_params__", "__parameters__")
"""Where a class records the type parameters it declares.

A PEP 695 declaration records them under ``__type_params__``; the legacy
``Generic[T]`` and ``Protocol[T]`` spellings record them under ``__parameters__``.
A schema is answered by whichever the declaration used.
"""


def _declared_parameters(klass: type[Any]) -> tuple[Any, ...]:
    """Read the type parameters one class declares in its own namespace.

    The value is read raw and required to be a tuple, since ``type`` itself binds
    ``__type_params__`` as the descriptor that serves every class rather than as
    parameters of its own.

    Args:
      klass (type[Any]): The class to inspect.

    Returns:
      tuple[Any, ...]: The parameters it declares, empty when it declares none.
    """
    namespace = type.__getattribute__(klass, "__dict__")
    for attribute in _PARAMETER_ATTRIBUTES:
        parameters = namespace.get(attribute, ())
        if isinstance(parameters, tuple) and len(parameters) > 0:
            return parameters
    # A parameterized alias such as list[T] contributes its origin as the base and
    # keeps its parameters on the alias, which the created class records only in
    # __orig_bases__. A concrete alias such as dict[str, int] holds none.
    bases = namespace.get("__orig_bases__", ())
    if isinstance(bases, tuple):
        for base in bases:
            parameters = getattr(base, "__parameters__", ())
            if isinstance(parameters, tuple) and len(parameters) > 0:
                return parameters
    return ()


def _type_parameter_owner(config_cls: type[Any]) -> tuple[type[Any], tuple[Any, ...], type[Any] | None] | None:
    """Find the class that owns the type parameters a schema would carry.

    The schema's own MRO answers first, so a section inheriting from a generic
    base is reported against that base. The metaclass MRO answers after it, since
    a generic metaclass can introduce its parameter into the annotations the class
    it builds carries.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      tuple[type[Any], tuple[Any, ...], type[Any] | None] | None: The declaring
        class, its parameters, and the schema it is the metaclass of when it was
        found that way; None when nothing in either chain declares any.
    """
    for klass in type.__getattribute__(config_cls, "__mro__"):
        parameters = _declared_parameters(klass)
        if len(parameters) > 0:
            return klass, parameters, None
    for klass in type.__getattribute__(type(config_cls), "__mro__"):
        parameters = _declared_parameters(klass)
        if len(parameters) > 0:
            return klass, parameters, config_cls
    return None


def type_parameter_message(owner: type[Any], parameters: tuple[Any, ...], *, schema: type[Any] | None = None) -> str:
    """Build the rejection message for a schema class taking type parameters.

    Args:
      owner (type[Any]): The class declaring the parameters.
      parameters (tuple[Any, ...]): The parameters it declares.
      schema (type[Any] | None = None): The schema class the owner is the metaclass
        of, when the owner was found through the metaclass chain.

    Returns:
      str: The rejection message naming the parameters and the remedy that fits
        where the parameters were found.
    """
    names = ", ".join(class_label(parameter) for parameter in parameters)
    plural = "" if len(parameters) == 1 else "s"
    reason = "a config schema names the concrete types a file carries, since a load builds the type an annotation names"
    if schema is not None:
        return (
            f"{class_label(owner)}, the metaclass of {class_label(schema)}, takes the type parameter{plural} "
            f"{names}, and {reason}; build {class_label(schema)} with a metaclass that takes none"
        )
    return (
        f"{class_label(owner)} takes the type parameter{plural} {names}, and {reason}; declare the section with "
        f"those types written out, and derive anything that varies in an init=False field"
    )


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

    Membership is read from the raw class namespaces along the MRO, where
    ``dataclasses.is_dataclass`` reads it by attribute lookup. A class reaches
    this check merely by being named as an annotation, and raw reads keep that
    class's own code, including any ``__getattr__`` its metaclass defines, out of
    the walk, so whatever it would raise stays inside the issue collector.

    Args:
      hint (Any): The resolved type hint to inspect.

    Returns:
      bool: True when the hint is a dataclass type rather than an instance.
    """
    if not isinstance(hint, type):
        return False
    return _declared_in_namespaces(_class_namespaces(hint), "__dataclass_fields__")


class _HintKind(Enum):
    """The unmarshal dispatch category of a resolved type hint."""

    ANY = "any"
    NONE = "none"
    LITERAL = "literal"
    UNION = "union"
    CONTAINER = "container"
    UNSUPPORTED_GENERIC = "unsupported_generic"
    DATACLASS = "dataclass"
    CHOICE = "choice"
    CONFIG_VALUE = "config_value"
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
      choice_group (type[Any] | None): The variant-group base for a CHOICE hint.
        The group's membership is deliberately absent: it moves as modules
        import, and holding it here would pin one moment's registry into a cache
        keyed by the hint alone.
    """

    kind: _HintKind
    stripped: Any
    origin: Any
    args: tuple[Any, ...]
    dataclass_type: type[Any] | None = None
    choice_group: type[Any] | None = None


def _classify_hint_uncached(hint: Any) -> _HintClass:
    """Compute the dispatch structure of a resolved hint without caching.

    Args:
      hint (Any): A resolved type hint, possibly wrapped in ``Annotated``.

    Returns:
      _HintClass: The structural classification, mirroring the unmarshal engine's
        post-strip dispatch order.
    """
    stripped = _strip_annotated(hint)
    if stripped is ConfigValue or stripped is ConfigScalar:
        # Matched by identity ahead of any structural read, so the alias body is
        # the plain-data rule the walk applies rather than a shape to recurse into.
        return _HintClass(_HintKind.CONFIG_VALUE, stripped, None, ())
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
    if is_group(stripped):
        # A group base is a dataclass too, so it is matched ahead of the section
        # branch: the annotation names the set, and the file's tag names which
        # member of it to build.
        return _HintClass(_HintKind.CHOICE, stripped, None, (), choice_group=stripped)
    if isinstance(stripped, type) and _is_dataclass_type(stripped):
        return _HintClass(_HintKind.DATACLASS, stripped, None, (), dataclass_type=stripped)
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


@lru_cache(maxsize=_HINT_PLAN_CACHE_MAX)
def _array_match_by_id(key: _IdKey, backend: BackendSnapshot) -> _arrays.AnnotationMatch:
    """Cache-backed array classification keyed by hint identity and loaded backends.

    Args:
      key (_IdKey): Identity wrapper around the hint.
      backend (BackendSnapshot): The backends loaded for the operation, which is
        what the classification depends on beyond the hint itself.

    Returns:
      _arrays.AnnotationMatch: The classification for ``key.hint``.
    """
    return _arrays.inspect_annotation(key.hint, backend)


def array_match(hint: Any, backend: BackendSnapshot) -> _arrays.AnnotationMatch:
    """Classify a hint against the operation's backends, reusing a bounded cache.

    An array annotation names classes belonging to a loaded backend, so the
    answer is a function of the hint and of which backends are loaded, and the
    snapshot carries the second half into the key. That keeps the per-value walk
    from reclassifying one field's annotation once per element it holds, which is
    what a container of scalars would otherwise pay for every element.

    The snapshot is what the classification is made against as well as what it is
    stored under, so an entry describes the backends that produced it and a
    different set of them is a different entry.

    Args:
      hint (Any): A resolved type hint, with any ``Annotated`` wrapper intact.
      backend (BackendSnapshot): The operation's backend snapshot.

    Returns:
      _arrays.AnnotationMatch: The classification for ``hint``.
    """
    if _TYPE_CACHE_DISABLED:
        return _arrays.inspect_annotation(hint, backend)
    return _array_match_by_id(_IdKey(hint), backend)


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
      default_depth (int): Levels a direct ``field(default=...)`` value's plain
        form writes, measured once here so the build that selects it spends the
        nesting budget without walking the value again. Zero for a field with no
        direct default.
    """

    definition: Field[Any]
    has_default: bool
    conflicts: tuple[str, ...]
    default_depth: int = 0


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
    cached = _cached_on(config_cls, _FIELD_SLOT)
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
                default_depth=0 if field.default is MISSING else encoded_depth(field.default),
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
    _store_on(config_cls, _FIELD_SLOT, result)
    return result


def _validate_schema(config_cls: type[Any]) -> tuple[ConfigIssue, ...]:
    """Validate a dataclass's annotations and authored defaults against the supported set.

    This inspects the schema itself, independent of any config data, so an
    unsupported annotation is reported even when the field is omitted and falls
    back to its default, and every authored ``field(default=...)`` is judged
    whether or not the input supplies its field.

    Annotation issues and default issues accumulate separately during the walk and
    join at the end. Keeping them apart is what lets one class's bad default leave
    an enclosing class's default still checked, while an unsupported annotation
    still suppresses the default that would only restate it.

    Args:
      config_cls (type[Any]): The entry dataclass to validate.

    Returns:
      tuple[ConfigIssue, ...]: The schema issues found, annotation issues first,
        empty when the schema is fully supported.
    """
    # Read before the walk and stored with the result: a walk that reached a
    # variant group saw this membership, so a later registration marks it stale.
    generation = registry_generation()
    cached = _cached_on(config_cls, _SCHEMA_SLOT, generation=generation)
    if cached is not None:
        return cached
    issues: list[ConfigIssue] = []
    defaults: list[ConfigIssue] = []
    entry_message = _entry_type_message(config_cls)
    if entry_message is None:
        if is_group(config_cls):
            # A group reached as the entry class is the same annotation a field
            # would name, so the walk that covers its variants runs here too.
            _validate_choice_schema(config_cls, "", issues, defaults, set())
        else:
            _validate_dataclass_schema(config_cls, "", issues, defaults, set())
    else:
        # Field classification reads dataclasses.fields, so a non-dataclass entry
        # is reported here rather than walked.
        issues.append(ConfigIssue("", entry_message))
    result = (*issues, *defaults)
    if registry_generation() != generation:
        # A variant registered while this walk ran, which the walk may have
        # passed before it arrived. The result describes a registry that no
        # longer holds, so it is recomputed rather than returned or cached.
        return _validate_schema(config_cls)
    _store_on(config_cls, _SCHEMA_SLOT, result, generation=generation)
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
    return missing_dataclass_message(config_cls)


def missing_dataclass_message(config_cls: Any) -> str:
    """Build the rejection message for a class carrying no config schema.

    Shared by the entry route and the nested-field route so a class named as a
    section reads the same way wherever it turns up.

    Args:
      config_cls (Any): The class that carries no schema.

    Returns:
      str: The rejection message naming the class and the required remedy.
    """
    label = class_label(config_cls)
    return f"{label} is not a dataclass, so it {_MISSING_DATACLASS_MARKER}. Declare it with @dataclass."


_SUPPORTED_ANNOTATIONS = (
    "bool, int, float, str, Path, date/time, Enum/Literal, dataclass, container/union, array/tensor, "
    "or ConfigValue/ConfigScalar for plain data"
)
"""The supported annotation categories, named in every type-boundary message."""


def unsupported_hint_message(hint: Any) -> str:
    """Build the message for an annotation outside the supported type set.

    A class that looks like a schema whose declaration skipped ``@dataclass``
    routes to the decorator remedy instead, since naming the supported categories
    would answer a question the author did not ask.

    Args:
      hint (Any): The resolved type hint that has no supported handling.

    Returns:
      str: Either the missing-decorator message or the type-boundary message.
    """
    if _looks_like_undecorated_schema(hint):
        return missing_dataclass_message(hint)
    return (
        f"{_UNSUPPORTED_PREFIX}{_hint_name(hint)}; choose a supported annotation "
        f"({_SUPPORTED_ANNOTATIONS}) and derive other runtime values in an init=False field"
    )


def open_data_message(hint: Any) -> str:
    """Build the rejection message for an annotation that names no element type.

    ``Any`` and an argument-free container each leave the values they hold
    undescribed, so each is answered with the alias that names confingo's
    plain-data domain along with the parameterized form to write.

    Args:
      hint (Any): The resolved annotation that leaves its contents undescribed.

    Returns:
      str: The rejection message naming the remedy.
    """
    if hint is Any:
        return (
            "Any leaves the values it holds undescribed; annotate the field ConfigValue "
            "(from confingo) for plain data of any shape, or name the type the field holds"
        )
    written = _BARE_CONTAINER_FORMS.get(hint, f"{_hint_name(hint)}[ConfigValue]")
    return (
        f"{_hint_name(hint)} carries no element type; write {written} for plain data of any shape, "
        f"or name the element type"
    )


MAX_PLAIN_DEPTH = 64
"""Deepest nesting any walk over plain data follows before it reports and stops.

Every walk confingo runs over a value -- coercion, authored-default validation,
serialization, and equality -- carries this budget alongside an identity stack.
The stack ends a structure that closes back on itself at the value that closes
it, and the budget ends one that is merely deeper than a walk supports, so both
arrive as a path-tagged issue.
"""


MAX_RENDER_HOPS = 64
"""How many times a walk follows one array's plain form into another before stopping.

``tolist`` belongs to an array's own class, and a class may answer with another
array whose own ``tolist`` answers with a third. An identity stack ends a chain
that hands back an array the walk already holds; this budget ends one that hands
back a new array every time, so a chain that never reaches a plain form arrives
as a path-tagged issue rather than as a raw ``RecursionError``. It is counted
apart from the nesting budget, since a render hop writes no container level.
"""


def encoded_depth(
    value: Any,
    seen: tuple[int, ...] = (),
    budget: int = MAX_PLAIN_DEPTH + 1,
    renders: int = MAX_RENDER_HOPS,
) -> int:
    """Count the container levels a value's plain form writes.

    Counts what the marshal walk counts, so a value measured here is charged the
    levels that walk spends on it. A scalar writes no level of its own; a
    container writes one plus the deepest level its contents write; a section
    writes one plus the deepest its exported fields write, leaving ``init=False``
    runtime state out exactly as serialization does; and an array writes one level
    per axis up to the first empty one, since an empty axis ends the encoding. An
    array class supplying its own plain form is rendered once and the result
    walked, since what such a class writes is settled by what it hands back rather
    than by the shape it carries.

    The walk is bounded the way every other walk over a value is. A value that
    reaches itself, or one nested past the budget, is answered as deeper than the
    budget allows rather than followed, so the measure terminates and the
    authored-value validator is still the thing that reports it at its own path.

    Args:
      value (Any): The value to measure.
      seen (tuple[int, ...] = ()): Ids of the containers open on this branch.
      budget (int = MAX_PLAIN_DEPTH + 1): Levels left before the answer is capped.
      renders (int = MAX_RENDER_HOPS): Array-into-array render hops left.

    Returns:
      int: The levels the value's plain form writes, capped one past the budget.
    """
    beyond = MAX_PLAIN_DEPTH + 1
    rank = _arrays.encoded_array_depth(value)
    if rank is not None:
        return rank
    rendered = _arrays.render_own_plain_form(value)
    if rendered is _arrays.FAILED:
        # The value matched an array form and declined to render, so it writes no
        # plain form to measure; the authored-value validator reports it at its
        # own path, and charging it nothing here leaves that report the only one.
        return 0
    if rendered is not _arrays.NOT_ARRAY:
        # The rendered form comes from the value's own class, which can hand back a
        # graph that reaches the array again, so the array joins the branch before
        # the walk follows what it produced. A render hop writes no container
        # level, so it spends the hop budget rather than the nesting one.
        if not _arrays.is_array_value(rendered):
            # The class reached lists and numbers, so this render followed into no
            # further array and the measure carries on with the hops it holds.
            return encoded_depth(rendered, (*seen, id(value)), budget, renders)
        if renders <= 0 or id(value) in seen:
            # The chain writes no plain form to measure. The authored-value
            # validator walks it and names what it found, so charging nothing
            # here leaves that report the only one.
            return 0
        return encoded_depth(rendered, (*seen, id(value)), budget, renders - 1)
    if isinstance(value, Mapping):
        children: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif _is_dataclass_type(type(value)) and not isinstance(value, type):
        exported = _classify_dataclass_fields(type(value)).init_fields
        children = (getattr(value, item.definition.name, None) for item in exported)
    else:
        return 0
    if budget <= 0 or id(value) in seen:
        return beyond
    branch = (*seen, id(value))
    deepest = max((encoded_depth(child, branch, budget - 1, renders) for child in children), default=0)
    return min(1 + deepest, beyond)


def plain_scalar_message(value: Any) -> str:
    """Build the rejection message for a value outside the ``ConfigScalar`` domain.

    Args:
      value (Any): The value that carries no single plain leaf form.

    Returns:
      str: The rejection message naming the type found and the domain required.
    """
    return f"expected one plain scalar for ConfigScalar, got {_typename(value)}"


def plain_data_message(value: Any) -> str:
    """Build the rejection message for a value outside the ``ConfigValue`` domain.

    Args:
      value (Any): The value that carries no plain form.

    Returns:
      str: The rejection message naming the type found and the shapes accepted.
    """
    return (
        f"expected plain data for ConfigValue, got {_typename(value)}; "
        f"use a scalar, a list, or a str-keyed mapping, or name the type with a dataclass section"
    )


def plain_key_message(key: Any) -> str:
    """Build the rejection message for a mapping key outside the plain-data domain.

    The key is named by its type rather than its text, so a key whose ``__str__``
    raises is reported instead of run.

    Args:
      key (Any): The mapping key that is not a ``str``.

    Returns:
      str: The rejection message naming the type found.
    """
    return f"expected a str mapping key, got {_typename(key)}"


def plain_cycle_message() -> str:
    """Build the rejection message for a value that reaches itself.

    Returns:
      str: The rejection message naming the remedy.
    """
    return "value holds itself, so it has no plain form; supply a structure that terminates"


def render_hop_message() -> str:
    """Build the rejection message for a render chain that keeps producing arrays.

    Returns:
      str: The rejection message naming the limit and the remedy.
    """
    return (
        f"rendering the plain form followed {MAX_RENDER_HOPS} arrays into one another; "
        f"answer with lists and numbers from tolist so the plain form terminates"
    )


def plain_depth_message() -> str:
    """Build the rejection message for a value nested past the walk's budget.

    Returns:
      str: The rejection message naming the limit and both remedies.
    """
    return (
        f"nesting reaches the {MAX_PLAIN_DEPTH} level limit for plain data; "
        f"flatten the structure, or name the shape with a dataclass section"
    )


_BARE_CONTAINER_FORMS: dict[Any, str] = {
    tuple: "tuple[ConfigValue, ...]",
    list: "list[ConfigValue]",
    set: "set[ConfigScalar]",
    frozenset: "frozenset[ConfigScalar]",
    dict: "dict[str, ConfigValue]",
    Sequence: "Sequence[ConfigValue]",
    Mapping: "Mapping[str, ConfigValue]",
}
"""Each argument-free container annotation mapped to the form that names its contents."""


_UNSUPPORTED_PREFIX = "unsupported field type "
"""Opening text of the type-boundary message, shared by its builder and its readers."""

_MISSING_DATACLASS_MARKER = "carries no config schema"
"""Text the missing-decorator message carries, shared by its builder and its readers."""


_FOREIGN_MODEL_MARKERS: tuple[str, ...] = ("__attrs_attrs__", "model_fields", "__pydantic_fields__")
"""Attributes marking a class that already belongs to another schema system.

Membership is read by name so neither library has to be installed for the check
to run, and a class carrying one is left on the type-boundary message: its
annotations are deliberate declarations to that system rather than a config
schema missing its decorator.
"""

_CLASS_VAR_TEXT = re.compile(r"^\s*(?:\w+\s*\.\s*)*ClassVar\s*(?:\[|$)")
"""Matches a ``ClassVar`` annotation written as source text, qualified or bare."""


def _looks_like_undecorated_schema(hint: Any) -> bool:
    """Report whether a class declares schema-shaped annotations without being a dataclass.

    The signal is a class that carries its own non-``ClassVar`` annotations and no
    dataclass fields, which is what a section reads like when its declaration
    skipped the decorator. Classes whose annotations are deliberate declarations
    to something else are excluded: ``TypedDict`` and ``NamedTuple``, protocols,
    generic classes carrying type parameters, and models belonging to another
    schema system.

    Every fact is read from the raw class namespaces along the MRO, and
    annotations are read as declared rather than resolved. Naming a class as a
    field type is the only thing the author did, so classifying it runs none of
    its code: resolution would evaluate its annotation expressions, and ordinary
    attribute lookup would invoke whatever ``__getattr__`` or ``__getattribute__``
    its metaclass defines. Either would let an arbitrary exception escape the
    issue collector.

    Args:
      hint (Any): The resolved type hint that has no supported handling.

    Returns:
      bool: True when the remedy is to decorate the class with ``@dataclass``.
    """
    if not isinstance(hint, type):
        return False
    if typing.is_typeddict(hint):
        return False
    namespaces = _class_namespaces(hint)
    if _declared_in_namespaces(namespaces, "__dataclass_fields__"):
        return False
    if _is_named_tuple(hint, namespaces):
        return False
    if any(namespace.get("_is_protocol", False) is True for namespace in namespaces):
        return False
    if any(len(namespace.get("__parameters__", ())) > 0 for namespace in namespaces):
        return False
    if any(_declared_in_namespaces(namespaces, marker) for marker in _FOREIGN_MODEL_MARKERS):
        return False
    own = own_annotations(hint)
    return any(not _declares_class_var(annotation) for annotation in own.values())


def _is_named_tuple(hint: type[Any], namespaces: tuple[Mapping[str, Any], ...]) -> bool:
    """Report whether a class is a ``NamedTuple``, read without attribute lookup.

    ``_fields`` carries the role for a ``NamedTuple`` that ``__dataclass_fields__``
    carries for a dataclass, but the name is ordinary and any class may bind it,
    so the tuple base is required alongside it.

    Args:
      hint (type[Any]): The class to inspect.
      namespaces (tuple[Mapping[str, Any], ...]): Its raw MRO namespaces.

    Returns:
      bool: True when the class is a ``NamedTuple`` subclass.
    """
    if not _declared_in_namespaces(namespaces, "_fields"):
        return False
    return tuple in type.__getattribute__(hint, "__mro__")


def _class_namespaces(config_cls: type[Any]) -> tuple[Mapping[str, Any], ...]:
    """Read a class's raw namespaces along its MRO, most derived first.

    ``type.__getattribute__`` is used rather than ordinary attribute access, so a
    ``__getattribute__`` the inspected class's metaclass defines stays out of the
    read. A metaclass is still what supplies ``__mro__`` itself, so a data
    descriptor bound under that name answers here as it answers anywhere.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      tuple[Mapping[str, Any], ...]: One namespace per class in the MRO.
    """
    mro = type.__getattribute__(config_cls, "__mro__")
    return tuple(type.__getattribute__(klass, "__dict__") for klass in mro)


if sys.version_info >= (3, 14):
    import annotationlib

    def own_annotations(config_cls: type[Any]) -> Mapping[str, Any]:
        """Read the annotations a class declares itself, left unevaluated.

        The string format is what keeps the read total: a schema is inspected
        while its own class statement is still running, so an annotation naming
        a class defined further down the module resolves only after that
        statement finishes.

        Args:
          config_cls (type[Any]): The class to inspect.

        Returns:
          Mapping[str, Any]: The class's own annotations, each value the source
            text of the declaration.
        """
        return annotationlib.get_annotations(config_cls, format=annotationlib.Format.STRING)

else:

    def own_annotations(config_cls: type[Any]) -> Mapping[str, Any]:
        """Read the annotations a class declares itself, left unevaluated.

        The class namespace holds the declarations, so reading it directly keeps
        inherited annotations out and evaluates nothing.

        Args:
          config_cls (type[Any]): The class to inspect.

        Returns:
          Mapping[str, Any]: The class's own annotations, each value a type
            object or the source text under postponed evaluation.
        """
        return config_cls.__dict__.get("__annotations__", {})


def _declared_in_namespaces(namespaces: tuple[Mapping[str, Any], ...], name: str) -> bool:
    """Report whether any namespace along an MRO binds ``name``.

    Args:
      namespaces (tuple[Mapping[str, Any], ...]): Raw namespaces along the MRO.
      name (str): The attribute name to look for.

    Returns:
      bool: True when some class in the MRO binds the name.
    """
    return any(name in namespace for namespace in namespaces)


def _declares_class_var(annotation: Any) -> bool:
    """Report whether one declared annotation names ``ClassVar``, unevaluated.

    Args:
      annotation (Any): An entry from a class's own ``__annotations__``, which is
        a string under postponed evaluation and a type object otherwise.

    Returns:
      bool: True when the declaration is a ``ClassVar``.
    """
    if isinstance(annotation, str):
        return _CLASS_VAR_TEXT.match(annotation) is not None
    return _is_class_var(annotation)


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
    own = own_annotations(config_cls)
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


def _facade_collision_messages(config_cls: type[Any]) -> list[str]:
    """Collect the reserved-name collisions a node carries, read at preflight.

    A node is checked again here because a class built programmatically receives
    its annotations once its creation has finished, which is after the check that
    runs from ``__init_subclass__``. Preflight sees the completed class, so the
    collision is named the same way whichever route declared it.

    The import is deferred because ``_node`` sits above this module
    (``_node`` -> ``_core`` -> ``_schema``).

    Args:
      config_cls (type[Any]): The class being validated.

    Returns:
      list[str]: One message per shadowed name, empty when the facade is intact.
    """
    from confingo._node import (  # noqa: PLC0415
        _facade_collisions,
        _is_config_node,
    )

    return _facade_collisions(config_cls) if _is_config_node(config_cls) else []


def _is_class_var(hint: Any) -> bool:
    """Report whether a resolved hint declares a ``ClassVar``.

    Args:
      hint (Any): The resolved type hint to inspect.

    Returns:
      bool: True when the hint is ``ClassVar`` or a ``ClassVar[...]`` subscription.
    """
    return hint is ClassVar or get_origin(hint) is ClassVar


def _validate_dataclass_schema(
    config_cls: type[Any],
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
    """Collect schema issues for one dataclass, recursing into nested dataclasses.

    Args:
      config_cls (type[Any]): The dataclass to inspect.
      path (str): Dotted schema path of this node, empty at the root.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      defaults (list[ConfigIssue]): Destination for authored-default issues, kept
        apart from annotation issues and joined at the end of the walk.
      seen (set[type[Any]]): Dataclasses already visited on this path, to break reference cycles.
    """
    if config_cls in seen:
        return
    seen = seen | {config_cls}
    hints = _resolved_hints(config_cls)
    undecorated = _undecorated_node_message(config_cls, hints)
    if undecorated is not None:
        issues.append(ConfigIssue(path, undecorated))
    issues.extend(
        ConfigIssue(path, message)
        for message in (_dispatch_protocol_message(config_cls), _constructor_message(config_cls))
        if message is not None
    )
    issues.extend(ConfigIssue(path, message) for message in _facade_collision_messages(config_cls))
    # Checked on every class carrying a selection key rather than only through
    # the group, so a variant named directly by a field, or reached as the entry
    # class, is held to it too. A field of that name would overwrite the
    # selection the marshal writes, leaving a section that rejects its own output.
    collision = _selection_key_collision_message(config_cls)
    if collision is not None:
        issues.append(ConfigIssue(path, collision))
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
        _validate_hint_schema(hints[field.name], field_path, issues, defaults, seen)
        if len(issues) == before and field.default is not MISSING:
            # The annotation holds up, so the authored default can be judged
            # against it. The value already exists, so reading it runs no
            # user code on this cached path; a default_factory is left to the
            # one build that selects it.
            _validate_direct_default(field.default, hints[field.name], field_path, defaults)


_DISPATCH_PROTOCOLS: tuple[tuple[Any, str], ...] = (
    (Mapping, "a mapping"),
    (Sequence, "a sequence"),
    (set, "a set"),
    (frozenset, "a frozenset"),
    (Enum, "an enum"),
    (Path, "a path"),
    (dt.date, "a date"),
    (dt.time, "a time"),
)
"""The kinds every walk over a value dispatches on, paired with how a message names one.

A section is recognized by being a dataclass and each of these by being an
instance of the kind, so a class that is both answers to two readings of one
value. Which reading a walk takes is then decided by the order that walk happens
to test in, which is no contract to build a config format on, so the shape is
named at preflight instead.
"""


def _dispatch_protocol_message(config_cls: type[Any]) -> str | None:
    """Report the walk-dispatch kind a schema class also is, or None for a plain section.

    Args:
      config_cls (type[Any]): The dataclass being validated.

    Returns:
      str | None: The rejection message naming the kind and the remedy, or None.
    """
    for protocol, described in _DISPATCH_PROTOCOLS:
        if not issubclass(config_cls, protocol):
            continue
        kind = described.split(" ", 1)[1]
        return (
            f"{class_label(config_cls)} is a config section and also {described}, and every walk over a value "
            f"reads those as two different things, so what the section writes and what a file rebuilds it as "
            f"follow from which reading a walk reaches first; declare the section as a plain dataclass, and "
            f"carry the {kind} on an object held in an init=False field"
        )
    return None


def _constructor_message(config_cls: type[Any]) -> str | None:
    """Report why a class's constructor cannot be handed its own fields, or None.

    confingo builds a config object by calling the class with its ``init=True``
    field names, so the constructor has to accept exactly that call: every one of
    those names, and nothing further that it requires. The generated ``__init__``
    satisfies this by construction. An initializer the author bound in its place
    need not, and neither does the generated one once a required ``InitVar`` adds
    a parameter a config file has no way to supply.

    Reading the signature answers the question the build asks, so this rests on
    what the constructor accepts rather than on telling a generated body from an
    authored one.

    Args:
      config_cls (type[Any]): The dataclass being validated.

    Returns:
      str | None: The rejection message naming what the call cannot supply, or
        None when the constructor accepts the class's own fields.
    """
    initializer = _resolve_constructor(config_cls)
    remedy = (
        "confingo builds a config object by calling the class with its field names; leave the generated "
        "__init__ in place, and derive anything else in __post_init__ or an init=False field"
    )
    if not inspect.isfunction(initializer):
        return f"{class_label(config_cls)} carries an __init__ that is not a Python function, and {remedy}"
    try:
        signature = inspect.signature(initializer)
    except (TypeError, ValueError):
        return f"{class_label(config_cls)} carries an __init__ whose signature cannot be read, and {remedy}"
    parameters = list(signature.parameters.values())[1:]
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    takes_any_keyword = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters)
    accepted = {item.name for item in parameters if item.kind in keyword_kinds}
    field_names = [item.name for item in fields(config_cls) if item.init]

    unreachable = [name for name in field_names if name not in accepted]
    if len(unreachable) > 0 and not takes_any_keyword:
        named = ", ".join(unreachable)
        noun = "argument" if len(unreachable) == 1 else "arguments"
        return f"{class_label(config_cls)}.__init__ takes no {named} {noun} for the field of that name, and {remedy}"

    supplied = set(field_names)
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    demanded = [
        item.name
        for item in parameters
        if item.default is inspect.Parameter.empty and item.kind not in variadic and item.name not in supplied
    ]
    if len(demanded) > 0:
        named = ", ".join(demanded)
        noun = "argument" if len(demanded) == 1 else "arguments"
        return (
            f"{class_label(config_cls)}.__init__ requires the {named} {noun}, which names no field a config "
            f"file can supply; give it a default, or declare it as an ordinary field when the file carries "
            f"the value"
        )
    return None


def _resolve_constructor(config_cls: type[Any]) -> Any:
    """Find the ``__init__`` a call on this class would reach.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      Any: The initializer bound nearest along the MRO.
    """
    for klass in type.__getattribute__(config_cls, "__mro__"):
        namespace = type.__getattribute__(klass, "__dict__")
        if "__init__" in namespace:
            return namespace["__init__"]
    return object.__init__


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
    # The schema path names one level of the plain document per segment, so a
    # default is judged under the nesting budget the whole-config walks reach it
    # with rather than a fresh one.
    validate_authored_value(value, hint, path, collector, label=DEFAULT_LABEL, depth=path.count(".") + 1)
    issues.extend(collector.issues)


def _validate_hint_schema(
    hint: Any,
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
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
    if hint is ConfigValue or hint is ConfigScalar or hint is type(None):
        return
    if hint is Any:
        issues.append(ConfigIssue(path, open_data_message(hint)))
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
        _validate_union_schema(hint, args, path, issues, defaults, seen)
        return

    if origin is not None:
        _validate_generic_schema(hint, origin, args, path, issues, defaults, seen)
        return

    if is_group(hint):
        _validate_choice_schema(hint, path, issues, defaults, seen)
        return

    if _is_dataclass_type(hint):
        _validate_dataclass_schema(hint, path, issues, defaults, seen)
        return

    if isinstance(hint, type):
        if hint in _BARE_CONTAINER_FORMS:
            issues.append(ConfigIssue(path, open_data_message(hint)))
            return
        if issubclass(hint, Enum):
            message = enum_schema_message(hint)
            if message is not None:
                issues.append(ConfigIssue(path, message))
            return
        if hint in _EXACT_SCALAR_TYPES:
            return
        if issubclass(hint, (Path, dt.date, dt.time)):
            issues.append(ConfigIssue(path, scalar_subclass_message(hint)))
            return

    issues.append(ConfigIssue(path, unsupported_hint_message(hint)))


def _validate_generic_schema(
    hint: Any,
    origin: Any,
    args: tuple[Any, ...],
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
    """Collect schema issues for a parameterized generic annotation.

    Args:
      hint (Any): The whole generic hint.
      origin (Any): ``get_origin(hint)``, the container or other generic origin.
      args (tuple[Any, ...]): ``get_args(hint)``, the arguments it was written with.
      path (str): Dotted schema path of the field carrying this hint.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      defaults (list[ConfigIssue]): Destination for authored-default issues.
      seen (set[type[Any]]): Dataclasses already visited on this path.
    """
    if origin in _BARE_CONTAINER_FORMS and not hasattr(hint, "__args__"):
        # A legacy typing alias written without arguments spells the same
        # argument-free container the builtin does, so it reads the same remedy.
        # Carrying no ``__args__`` at all is what marks it, which is how it
        # stays distinct from the explicit empty tuple ``tuple[()]``.
        issues.append(ConfigIssue(path, open_data_message(origin)))
        return
    if origin not in _CONTAINER_ORIGINS:
        issues.append(ConfigIssue(path, unsupported_hint_message(hint)))
        return
    malformed = _malformed_specialization(hint, origin, args)
    if malformed is not None:
        issues.append(ConfigIssue(path, malformed))
        return
    if origin in (dict, Mapping):
        key_hint = args[0] if len(args) == 2 else str
        value_hint = args[1] if len(args) == 2 else Any
        if key_hint is not str:
            message = f"unsupported dict key type {_hint_name(key_hint)}; only str keys are supported"
            issues.append(ConfigIssue(path, message))
        _validate_hint_schema(value_hint, path, issues, defaults, seen)
        return
    if origin in (set, frozenset):
        # An argument-free set carries Any elements, the same as a bare one.
        element_hints = args if len(args) > 0 else (Any,)
        for element_hint in element_hints:
            if element_hint is Ellipsis:
                continue
            _validate_set_element(hint, element_hint, path, issues, defaults, seen)
        return
    for element_hint in args:
        if element_hint is not Ellipsis:
            _validate_hint_schema(element_hint, path, issues, defaults, seen)


def _validate_union_schema(
    hint: Any,
    args: tuple[Any, ...],
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
    """Collect schema issues for a union, and hold it to one config section.

    A file gives one mapping for the field, so two sections in one union leave
    which of them it names to be decided by trying each in turn. A variant group
    is the annotation that settles it, and it counts as the union's one section
    however many variants stand behind it.

    Args:
      hint (Any): The whole union hint.
      args (tuple[Any, ...]): The union's members.
      path (str): Dotted schema path of the field carrying this hint.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      defaults (list[ConfigIssue]): Destination for authored-default issues.
      seen (set[type[Any]]): Dataclasses already visited on this path.
    """
    sections = [member for member in args if _names_section(member)]
    if len(sections) > 1:
        issues.append(ConfigIssue(path, _union_sections_message(hint, sections)))
    for member in args:
        _validate_hint_schema(member, path, issues, defaults, seen)


def _names_section(hint: Any) -> bool:
    """Report whether a union member names a config section.

    A variant group counts as one member however many variants stand behind it,
    since the group is the single annotation a file answers with one tagged
    mapping.

    Args:
      hint (Any): The resolved union member to inspect.

    Returns:
      bool: True when the member names a dataclass section or a variant group.
    """
    stripped = _strip_annotated(hint)
    return isinstance(stripped, type) and (is_group(stripped) or _is_dataclass_type(stripped))


def _union_sections_message(hint: Any, sections: list[Any]) -> str:
    """Build the rejection for a union naming more than one config section.

    Args:
      hint (Any): The whole union hint being rejected.
      sections (list[Any]): The members naming a section.

    Returns:
      str: The rejection naming the union, the count, and the group to declare.
    """
    named = ", ".join(_hint_name(section) for section in sections)
    return (
        f"{_hint_name(hint)} names {len(sections)} config sections in one union ({named}), and a config file "
        f"gives one mapping for the field, so which section it names is decided by trying each in turn. "
        f'Declare a variant group -- class Group(ConfigChoice, tag_key="..."), each section written as '
        f'class Section(Group, tag="...") -- and annotate the field with that group, so the mapping names '
        f"which section to build."
    )


def _validate_choice_schema(
    group: type[Any],
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
    """Collect schema issues for a variant group and every variant behind it.

    The group is the only class a field annotation names, so its variants are
    reachable for validation through the registry alone. Walking them here is
    what puts a variant's own annotations, defaults, and constructor under the
    same preflight the rest of the tree gets, ahead of any value being built.

    Each variant carries the group's fields too, so one walk per variant reaches
    a shared field once per variant; the issues this raises are deduplicated so
    the report names a shared problem once.

    Args:
      group (type[Any]): The variant-group base being validated.
      path (str): Dotted schema path of the field carrying the group.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      defaults (list[ConfigIssue]): Destination for authored-default issues.
      seen (set[type[Any]]): Dataclasses already visited on this path, to break
        reference cycles.
    """
    record = group_record(group)
    if record is None:
        return
    if group in seen:
        return
    if len(record.by_tag) == 0:
        issues.append(ConfigIssue(path, _empty_group_message(group)))

    # The group's own fields are walked once, and each variant inherits them, so
    # a variant's walk restates whatever the group's walk already reported. Only
    # those restatements are dropped: a field the variant declares itself is a
    # declaration of its own to fix, whatever the group says about the field of
    # that name, and two variants each declaring a defect are two fixes.
    shared: list[ConfigIssue] = []
    shared_defaults: list[ConfigIssue] = []
    _validate_dataclass_schema(group, path, shared, shared_defaults, seen)
    issues.extend(shared)
    defaults.extend(shared_defaults)
    inherited = {(issue.path, issue.message) for issue in (*shared, *shared_defaults)}

    for tag in sorted(record.by_tag):
        variant_cls = record.by_tag[tag]
        declared = set(own_annotations(variant_cls))
        variant: list[ConfigIssue] = []
        variant_defaults: list[ConfigIssue] = []
        _validate_dataclass_schema(variant_cls, path, variant, variant_defaults, seen)
        issues.extend(issue for issue in variant if _is_own_report(issue, path, declared, inherited))
        defaults.extend(issue for issue in variant_defaults if _is_own_report(issue, path, declared, inherited))


def _is_own_report(issue: ConfigIssue, path: str, declared: set[str], inherited: set[tuple[str, str]]) -> bool:
    """Report whether one variant issue says something the group's walk left unsaid.

    Args:
      issue (ConfigIssue): The issue the variant's walk produced.
      path (str): Dotted schema path of the field carrying the group.
      declared (set[str]): The field names the variant declares in its own body.
      inherited (set[tuple[str, str]]): Path and message of every issue the
        group's own walk already reported.

    Returns:
      bool: True when the issue belongs in the report.
    """
    if (issue.path, issue.message) not in inherited:
        return True
    relative = issue.path[len(path) :].lstrip(".")
    # A field the variant declares itself carries its own defect, so an identical
    # report from the group describes a second declaration rather than this one.
    return len(relative) > 0 and relative.split(".", 1)[0] in declared


def _empty_group_message(group: type[Any]) -> str:
    """Build the rejection for a variant group carrying no variants.

    Args:
      group (type[Any]): The group base being validated.

    Returns:
      str: The rejection naming the group and how to add a variant.
    """
    return (
        f"the variant group {class_label(group)} has no variants, so no config file can select one; write "
        f'class Section({group.__name__}, tag="...") for each section the group stands for, and import the '
        f"module declaring them alongside the schema that names the group."
    )


def _selection_key_collision_message(config_cls: type[Any]) -> str | None:
    """Report a field whose name is the key its group selects variants under.

    Args:
      config_cls (type[Any]): The class to inspect, a group base or a variant.

    Returns:
      str | None: The rejection naming both uses of the key, or None when the
        class belongs to no group or declares no field of that name.
    """
    group = config_cls if is_group(config_cls) else None
    if group is None:
        resolved = variant_tag(config_cls)
        if resolved is None:
            return None
        group = resolved[0]
    record = group_record(group)
    if record is None:
        return None
    tag_key = record.tag_key
    if tag_key not in _classify_dataclass_fields(config_cls).by_name:
        return None
    # The class named is the one whose body declares the field, so a field the
    # group declares reads the same from the group's walk and from every variant
    # that inherits it, which is what leaves one report for the one declaration
    # to fix.
    declaring = next(
        (base for base in config_cls.__mro__ if tag_key in own_annotations(base)),
        config_cls,
    )
    return (
        f"{class_label(declaring)} declares a field named {tag_key!r}, which is the key the variant group "
        f"{class_label(group)} selects a variant under, so one key in the section would name both. Rename "
        f"the field, or declare the group with a different tag_key=."
    )


_EXACT_SCALAR_TYPES: frozenset[Any] = frozenset({bool, int, float, str, Path, dt.datetime, dt.date, dt.time})
"""The scalar classes a field annotation names, matched exactly.

A load reads one of these from the text a file carries and builds the class named
here, so these are the classes an annotation can promise. A subclass names a class
confingo has no reading for: the annotation would be pronounced supported and then
produce a base-class value it does not describe, so preflight names the base to
write instead. ``Path`` stands for the platform class ``Path(...)`` resolves to,
which is an instance of ``Path`` on every platform.
"""


def scalar_subclass_message(hint: type) -> str:
    """Build the rejection message for a subclass of a supported scalar class.

    Args:
      hint (type): The subclass the annotation names.

    Returns:
      str: The rejection message naming the base to write and the remedy for the
        behavior the subclass carries.
    """
    base = next(candidate for candidate in (dt.datetime, dt.date, dt.time, Path) if issubclass(hint, candidate))
    return (
        f"{class_label(hint)} is a {base.__name__} subclass, and a load builds {base.__name__} itself; "
        f"annotate the field {base.__name__}, and derive the subclass in an init=False field"
    )


_EXACT_ENUM_VALUE_TYPES: tuple[type, ...] = (bool, int, str)
"""The classes an enum member value carries, matched exactly.

A load reads one of these from a file and looks the member up by that value, so a
member value has to be a class the lookup can be handed. Matching exactly is what
keeps the lookup single-valued: two members separated only by a subclass of one of
these write the same plain form, and the value a file carries rebuilds whichever
member the lookup reaches first.
"""


def enum_schema_message(hint: type[Enum]) -> str | None:
    """Build the rejection message for the first member value a load cannot rebuild.

    Args:
      hint (type[Enum]): The enum class the annotation names.

    Returns:
      str | None: The rejection message naming the value to write instead, or None
        when every member value carries an exact supported primitive type.
    """
    if type(hint).__call__ is not EnumType.__call__:
        return (
            f"enum {class_label(hint)} looks a member up through {class_label(type(hint))}, and a load rebuilds a "
            f"member by handing that lookup the value a file carries, so a member's own value can rebuild "
            f"as a different member; leave the lookup to EnumType, and map spellings outside the member "
            f"values in a _missing_ hook, which a lookup reaches after a member's own value resolves"
        )
    remedy = "give each member an exact bool, int, or str value"
    for member in hint:
        value = member.value
        if type(value) in _EXACT_ENUM_VALUE_TYPES:
            continue
        if isinstance(value, _EXACT_ENUM_VALUE_TYPES):
            base = next(candidate for candidate in _EXACT_ENUM_VALUE_TYPES if isinstance(value, candidate))
            return (
                f"enum {class_label(hint)} must carry primitive values; {member.name} is {value!r} of "
                f"{base.__name__} subclass {class_label(type(value))}, and a load rebuilds the exact "
                f"{base.__name__} a file carries, so two members separated only by that subclass "
                f"rebuild as one member; {remedy}"
            )
        return (
            f"enum {class_label(hint)} must carry primitive values; {member.name} is {value!r}, and a load "
            f"reads bool, int, or str from a file; {remedy}"
        )
    return None


def _malformed_specialization(hint: Any, origin: Any, args: tuple[Any, ...]) -> str | None:
    """Report how a container's type arguments depart from the form it builds from.

    Construction reads one element type from a sequence or set annotation and a
    key and value type from a mapping, and it reads ``...`` as the variadic marker
    of ``tuple[T, ...]``. An annotation carrying more arguments than that, or
    ``...`` anywhere else, describes something the engine has no reading for, and
    Python accepts the spelling at runtime, so preflight names it.

    Args:
      hint (Any): The resolved container hint as written.
      origin (Any): The container's unsubscripted origin type.
      args (tuple[Any, ...]): The container's type arguments.

    Returns:
      str | None: The rejection message, or None when the arguments fit the form.
    """
    ellipsis_positions = [index for index, arg in enumerate(args) if arg is Ellipsis]
    if origin is tuple:
        if len(ellipsis_positions) > 0 and (len(args) != 2 or ellipsis_positions != [1]):
            return (
                f"{_hint_name(hint)} carries ... outside the variadic form; write tuple[T, ...] for a "
                f"tuple of one repeated type, or name each position"
            )
        return None
    if len(ellipsis_positions) > 0:
        return (
            f"{_hint_name(hint)} carries ..., which marks a variadic tuple; name the element type, or "
            f"write tuple[T, ...] when a tuple of one repeated type is meant"
        )
    if len(args) == 0:
        # PEP 585 accepts an empty argument list on every origin, and only a tuple
        # reads it as a shape: ``tuple[()]`` names a tuple holding nothing. On any
        # other container it names no element type, which is what the
        # argument-free spelling names, so it reads the same remedy.
        return open_data_message(origin)
    expected = 2 if origin in (dict, Mapping) else 1
    if len(args) != expected:
        written = "type arguments" if expected == 2 else "type argument"
        return (
            f"{_hint_name(hint)} carries {len(args)} type arguments; {_hint_name(origin)} builds from "
            f"{expected} {written}"
        )
    return None


def _validate_set_element(
    hint: Any,
    element_hint: Any,
    path: str,
    issues: list[ConfigIssue],
    defaults: list[ConfigIssue],
    seen: set[type[Any]],
) -> None:
    """Collect the schema issues one ``set`` / ``frozenset`` element annotation carries.

    An annotation confingo has no construction semantics for is reported as the
    unsupported type it is, since what it would rebuild is undefined. A supported
    element that rebuilds a value a set cannot hold is reported against the
    container as written. A section carries the ``config_hash`` remedy, and its
    own schema is walked so defects inside it arrive in the same report.

    Args:
      hint (Any): The resolved ``set`` / ``frozenset`` hint being validated.
      element_hint (Any): The element annotation to inspect.
      path (str): Dotted schema path of the field carrying the container.
      issues (list[ConfigIssue]): Destination for any schema issues found.
      defaults (list[ConfigIssue]): Destination for authored-default issues.
      seen (set[type[Any]]): Dataclasses already visited on this path.
    """
    # The element's own walk runs first, and its findings reach the report
    # whatever else is decided, so a set element reports what the same annotation
    # reports as an ordinary field.
    element_issues: list[ConfigIssue] = []
    _validate_hint_schema(element_hint, path, element_issues, defaults, seen)
    issues.extend(element_issues)

    kind = _rebuild_kind(element_hint)
    if kind is _Rebuild.HASHABLE:
        union = _union_inside(element_hint)
        if union is not None:
            issues.append(ConfigIssue(path, _union_set_element_message(hint, union)))
        return
    if _holds_config_section(element_hint):
        # A section is unhashable whatever else its own schema reports, so the
        # remedy belongs beside the defects inside it.
        issues.append(ConfigIssue(path, _section_set_message(hint)))
        return
    if kind is _Rebuild.UNDECIDED:
        # The only evidence was an annotation confingo cannot build from, and its
        # own walk already reported that, so there is nothing further to claim.
        return
    issues.append(ConfigIssue(path, _unstable_set_element_message(hint, element_hint)))


class _Rebuild(Enum):
    """What a load rebuilds under one annotation, as far as the annotation says."""

    HASHABLE = "hashable"
    UNHASHABLE = "unhashable"
    UNDECIDED = "undecided"


def _rebuild_kind(hint: Any) -> _Rebuild:
    """Classify what a load rebuilds under an annotation.

    A set holds its elements by hash, and a load rebuilds every element from the
    plain form a file carries. Scalars rebuild as themselves, and ``tuple`` /
    ``frozenset`` rebuild as immutable containers exactly when their own arguments
    do, so the annotation settles it.

    The third answer is what keeps a report honest. An annotation confingo has no
    construction semantics for rebuilds something undecided, and a shape holding
    one inherits that only when nothing else in it already settles the question: a
    ``tuple`` holding a ``list`` rebuilds unhashable whatever sits beside the list.
    So a container reports its own instability when that instability is
    established on its own, and stays quiet when the only evidence was an
    annotation whose meaning is undefined.

    This covers the annotation, which is what a file round trip travels through.
    A value handed straight to ``from_dict`` may already be an instance of a
    supported type's subclass, and coercion returns such a value unchanged, so the
    object entering the set can carry hashing and equality of its own. Building
    the set is where that is answered, and it reports under the annotation as
    written.

    Args:
      hint (Any): The resolved element type hint to inspect.

    Returns:
      _Rebuild: HASHABLE when every rebuild is hashable, UNHASHABLE when the
        annotation establishes on its own that one is not, UNDECIDED when the
        annotation carries no construction semantics to decide from.
    """
    if _arrays.inspect_annotation(hint).matched:
        # An array rebuilds as an ordinary unhashable array.
        return _Rebuild.UNHASHABLE
    hint = _strip_annotated(hint)
    if hint is ConfigValue:
        # ConfigValue hands back whatever the plain form describes, including a list.
        return _Rebuild.UNHASHABLE
    if hint is ConfigScalar:
        # Every leaf of the scalar domain rebuilds hashable.
        return _Rebuild.HASHABLE
    if hint is Any:
        return _Rebuild.UNHASHABLE
    if hint is type(None):
        return _Rebuild.HASHABLE
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Literal:
        # Literal options are primitives, checked on their own above.
        return _Rebuild.HASHABLE
    if origin is typing.Union or origin is types.UnionType:
        return _combine(_rebuild_kind(member) for member in args)
    if origin is frozenset:
        # A frozenset hashes its own members while it is built, so a member that
        # rebuilds unhashable stops the frozenset from coming into existence
        # rather than producing an unhashable one. Every frozenset a load does
        # produce is hashable, and its members are judged on their own annotation
        # as the set elements they are.
        return _Rebuild.HASHABLE
    if origin is tuple:
        # A tuple holds whatever it is given, so a position that rebuilds
        # unhashable produces a tuple that a set cannot hold.
        return _combine(_rebuild_kind(arg) for arg in args if arg is not Ellipsis)
    if origin is not None:
        if origin in _CONTAINER_ORIGINS:
            # list, dict, set, Sequence, and Mapping all rebuild unhashable.
            return _Rebuild.UNHASHABLE
        # Another generic origin carries no construction semantics at all.
        return _Rebuild.UNDECIDED
    if _is_dataclass_type(hint):
        # A config object is unhashable by contract.
        return _Rebuild.UNHASHABLE
    if isinstance(hint, type):
        if hint in _BARE_CONTAINER_FORMS:
            # An argument-free container names no element type, and it still
            # rebuilds the container it names whatever those elements turn out to
            # be, so its own kind is settled: a frozenset hashes its members while
            # it is built, and every other container a load produces does not.
            return _Rebuild.HASHABLE if hint is frozenset else _Rebuild.UNHASHABLE
        if _is_supported_scalar(hint):
            return _Rebuild.HASHABLE if _hash_is_inherited(hint) else _Rebuild.UNHASHABLE
    return _Rebuild.UNDECIDED


def _combine(kinds: Iterable[_Rebuild]) -> _Rebuild:
    """Fold the answers for the parts of one shape into the answer for the shape.

    One part that rebuilds unhashable settles the shape whatever sits beside it,
    so it outranks a part whose meaning is undefined.

    Args:
      kinds (Iterable[_Rebuild]): One answer per part.

    Returns:
      _Rebuild: UNHASHABLE when any part is, UNDECIDED when any part is and none
        is unhashable, HASHABLE when every part is.
    """
    undecided = False
    for kind in kinds:
        if kind is _Rebuild.UNHASHABLE:
            return _Rebuild.UNHASHABLE
        if kind is _Rebuild.UNDECIDED:
            undecided = True
    return _Rebuild.UNDECIDED if undecided else _Rebuild.HASHABLE


def _union_inside(hint: Any) -> Any | None:
    """Find a union a set element names, at its own position or inside its shape.

    A set holds its elements by hash and equality while a file carries the plain
    form each was written as, and a union puts two readers behind one form: the
    load hands that form to the first member accepting it, which can rebuild an
    element the file did not carry or lose one it did. Deciding which unions are
    safe means proving what every member writes and what every member reads,
    which the annotation alone cannot settle for the types confingo supports, so
    a set element names one type instead.

    ``T | None`` is the exception, and the only one, because ``null`` is a plain
    form no other reader accepts: the two members can never be handed each
    other's form. The immutable shapes a set element can nest, ``tuple`` and
    ``frozenset``, carry the same rule in each of their positions.

    Args:
      hint (Any): The resolved element type hint to inspect.

    Returns:
      Any | None: The union found, or None when every position names one type.
    """
    stripped = _strip_annotated(hint)
    origin = get_origin(stripped)
    args = get_args(stripped)
    if origin is typing.Union or origin is types.UnionType:
        if len([arg for arg in args if arg is not type(None)]) > 1:
            return stripped
    elif origin is not tuple and origin is not frozenset:
        return None
    for arg in args:
        if arg is Ellipsis:
            continue
        found = _union_inside(arg)
        if found is not None:
            return found
    return None


def _union_set_element_message(hint: Any, union: Any) -> str:
    """Build the rejection message for a set element naming a union.

    Args:
      hint (Any): The resolved ``set`` / ``frozenset`` hint being rejected.
      union (Any): The union found at or inside the element.

    Returns:
      str: The rejection message naming the annotation, the union, and the remedies.
    """
    return (
        f"{_hint_name(hint)} cannot be built: a set element names one type, and {_hint_name(union)} "
        f"names several, so a load can hand one member's plain form to another and rebuild an element "
        f"the file did not carry; name the one type the elements carry, write T | None for an optional "
        f"element, or hold the values in a list"
    )


def _is_supported_scalar(hint: type) -> bool:
    """Report whether a class is one of the scalar types confingo builds.

    Args:
      hint (type): The class an element annotation names.

    Returns:
      bool: True when the class is a supported scalar leaf type.
    """
    if hint in (bool, int, float, str):
        return True
    return issubclass(hint, (Enum, Path, dt.date, dt.time))


def _hash_is_inherited(hint: type) -> bool:
    """Report whether a class leaves hashing to the implementation it inherits.

    Coercion returns a member of the annotated class itself, so a set holds
    whatever that class's ``__hash__`` describes. An enum may bind ``__hash__``
    to None, which makes its own members unhashable, or replace it with a method
    of its own. Reading the raw namespaces along the MRO answers this without
    calling anything, so a hash that raises is settled here rather than escaping
    a later construction.

    Args:
      hint (type): The scalar class an element annotation names.

    Returns:
      bool: True when the first ``__hash__`` along the MRO belongs to a class
        whose hashing confingo relies on.
    """
    mro = type.__getattribute__(hint, "__mro__")
    for klass in mro:
        namespace = type.__getattribute__(klass, "__dict__")
        if "__hash__" in namespace:
            return klass in _DEPENDABLE_HASH_OWNERS
    return True


_DEPENDABLE_HASH_OWNERS: frozenset[Any] = frozenset(
    klass
    for base in (object, Enum, bool, int, float, str, Path, dt.datetime, dt.date, dt.time)
    for klass in base.__mro__
    if "__hash__" in vars(klass)
)
"""Classes whose ``__hash__`` a set element may rely on.

Collected from the supported scalar types by walking each one's MRO, since the
implementation a supported type hashes by often belongs to a base: ``Path``
hashes by ``PurePath``, and ``IntEnum`` by ``int``. A scalar annotation reaching
one of these first along its own MRO hashes by the implementation confingo builds
its round trips against. A class binding ``__hash__`` itself describes hashing
confingo would have to call to learn, so it is settled at preflight instead.
"""


def _unstable_set_element_message(hint: Any, element_hint: Any) -> str:
    """Build the rejection message for a set element that rebuilds unhashable.

    Args:
      hint (Any): The resolved ``set`` / ``frozenset`` hint being rejected.
      element_hint (Any): The element annotation that fails to rebuild hashable.

    Returns:
      str: The rejection message naming the annotation, the element, and the remedies.
    """
    return (
        f"{_hint_name(hint)} cannot be built: a set element must rebuild hashable when a file is loaded, "
        f"and {_hint_name(element_hint)} rebuilds a value a set cannot hold; use a scalar element, a tuple "
        f"of scalars, or hold the values in a list"
    )


def _holds_config_section(hint: Any) -> bool:
    """Report whether a rejected set element names a config section.

    A config dataclass is unhashable, so an element annotation that names one --
    directly, as a union member, or inside the immutable ``tuple`` / ``frozenset``
    shapes that can themselves sit in a set -- has a remedy of its own worth
    naming: ``config_hash`` is the value-identity operation. This selects that
    message for an element ``_rebuild_kind`` has already rejected. The walk
    stops at the dataclass itself, so nothing recurses into its fields and a
    self-referential schema terminates.

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
        f"the collection, and use confingo.functional.config_hash(section) as the value-identity key when "
        f"uniqueness matters"
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
        base = class_label(origin)
        return f"{base}[{', '.join(_hint_name(arg) for arg in args)}]"
    return class_label(hint) if isinstance(hint, type) else str(getattr(hint, "__name__", hint))


def _typename(value: Any) -> str:
    """Name the runtime type of a value for error messages.

    Args:
      value (Any): The value to describe.

    Returns:
      str: The type name, with ``None`` reported as ``None``.
    """
    return "None" if value is None else class_label(type(value))


def _join(path: str, key: str) -> str:
    """Append a key to a dotted config path.

    Args:
      path (str): The parent dotted path, empty at the root.
      key (str): The child key or index to append.

    Returns:
      str: The combined dotted path.
    """
    return key if path == "" else f"{path}.{key}"
