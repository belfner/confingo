"""Canonical equality for config dataclasses.

Two configs are canonically equal exactly when their compared fields (``init=True``
and ``compare=True``) serialize to the same plain form. The engine here evaluates
that relation structurally: array and tensor fields compare through the backends'
vectorized operations wherever that comparison is provably exact, dataclass
sections and containers recurse over their compared fields, and every remaining
value pair compares by its plain-data COMPARE projection, so ``==`` runs at native
speed on large arrays while ``from_dict(cls, to_dict(config)) == config`` holds.
A ``compare=False`` field is carried by ``to_dict`` yet ignored here; an
``init=False`` runtime field is outside equality entirely.

Ordinary ``@dataclass`` declarations are the schema surface. A ``ConfigRoot``
subclass carries canonical ``__eq__`` from class-creation time (installed by
``ConfigRoot.__init_subclass__`` ahead of the ``@dataclass`` decorator, which
then skips generating its own), and every other schema dataclass receives it
at its first schema processing through ``_install_canonical_eq``. The
``config_equal`` free function exposes the same relation without touching the
classes involved.
"""

from __future__ import annotations

import math
from dataclasses import is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

from confingo import _arrays
from confingo._core import (
    _canonical_json,
    _classify_dataclass_fields,
    _PlainProjection,
    _project_plain,
)


if TYPE_CHECKING:
    import types

_MISSING = object()
"""Sentinel distinguishing an absent class-dict entry from a stored None."""

_CUSTOM_EQ_MARKER = "__confingo_custom_eq__"
"""Class attribute marking a ``ConfigRoot`` subclass whose body defines ``__eq__``.

``ConfigRoot.__init_subclass__`` stamps it when it finds a body-defined
``__eq__``, and ``_install_canonical_eq`` leaves marked classes untouched, so
a root's hand-written equality survives schema processing.
"""

_EXACT_PRIMITIVES = (bool, int, str)
"""Builtin scalar types whose own ``==`` matches canonical-JSON comparison exactly.

Membership is by exact type: subclasses such as ``np.float64`` (whose ``==``
applies NumPy promotion) and enum members (whose ``==`` may be overridden)
compare through their serialized forms instead. ``float`` is excluded because
``0.0 == -0.0`` while their canonical JSON differs (``0.0`` versus ``-0.0``), so
floats compare through the token-aware plain-form path to stay aligned with the
fingerprint.
"""


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two field values by their canonical serialized forms.

    Array pairs of the loaded backends compare through
    ``_arrays.native_equal`` where its vectorized path applies. Exact-type
    builtin scalars compare directly, dataclass pairs of the same class recurse
    over their compared fields (``init=True`` and ``compare=True``), sequence
    pairs and str-keyed dict pairs recurse structurally, and every other pair --
    scalar subclasses, enum members, sets, dicts with canonicalizing keys --
    compares by its COMPARE-projection plain form, which drops ``compare=False``
    fields from any dataclass reached that way. Each branch agrees with
    plain-form comparison on the supported value domain, so the walk is an
    evaluation strategy for one equality relation.

    Args:
        a: The left-hand value.
        b: The right-hand value.

    Returns:
        Whether the two values serialize to equal plain forms.
    """
    verdict = _arrays.native_equal(a, b)
    if verdict is not _arrays.NOT_COMPARABLE:
        return bool(verdict)
    if a is None or b is None:
        return a is b
    if type(a) is type(b) and type(a) in _EXACT_PRIMITIVES:
        # Same exact bool / int / str type, whose ``==`` matches the canonical
        # JSON tokens exactly. Cross-type pairs (``True`` vs ``1``) fall through
        # to the token-aware plain-form comparison below, so equality never
        # outruns the fingerprint.
        return bool(a == b)
    if type(a) is float and type(b) is float:
        # ``==`` matches the canonical JSON of finite floats except for signed
        # zero (``0.0`` and ``-0.0`` are equal but serialize to ``0.0`` / ``-0.0``),
        # which the sign check separates. Non-finite floats have no plain form, so
        # they compare by ``==`` here rather than raising in serialization.
        if a == 0.0 and b == 0.0:
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return bool(a == b)
    if is_dataclass(a) and not isinstance(a, type) and a.__class__ is b.__class__:
        compared = _classify_dataclass_fields(a.__class__).compared
        return all(_values_equal(getattr(a, c.definition.name), getattr(b, c.definition.name)) for c in compared)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
    if (
        isinstance(a, dict)
        and isinstance(b, dict)
        and all(type(key) is str for key in a)
        and all(type(key) is str for key in b)
    ):
        if a.keys() != b.keys():
            return False
        return all(_values_equal(item, b[key]) for key, item in a.items())
    left = _canonical_json(_project_plain(a, _PlainProjection.COMPARE))
    right = _canonical_json(_project_plain(b, _PlainProjection.COMPARE))
    return left == right


def _canonical_eq(self: Any, other: Any) -> bool | types.NotImplementedType:
    """Compare two config objects by canonical value equality.

    Equality ranges over the compared fields (``init=True`` and ``compare=True``):
    each compares through ``_values_equal``, which runs array fields through the
    backends' vectorized comparisons and recurses through sections and
    containers. ``init=False`` and ``compare=False`` fields carry no weight.

    Args:
        self: The left-hand config object.
        other: The right-hand operand.

    Returns:
        ``NotImplemented`` when ``other`` is a different class, else whether
        every compared field pair is canonically equal.
    """
    if other.__class__ is not self.__class__:
        return NotImplemented
    compared = _classify_dataclass_fields(self.__class__).compared
    return all(_values_equal(getattr(self, c.definition.name), getattr(other, c.definition.name)) for c in compared)


def _install_canonical_eq(config_cls: type[Any]) -> None:
    """Install canonical equality on a schema dataclass at schema processing.

    Every schema dataclass has ``_canonical_eq`` installed in place of the
    ``__eq__`` it carried, with a ``__hash__`` slot the dataclass set to
    None reverting to identity hashing, so sections and roots share one
    equality contract however they were declared. A ``ConfigRoot`` subclass
    already carries canonical equality from class-creation time, and one
    marked as defining its own ``__eq__`` keeps it.

    Args:
        config_cls: The schema dataclass being processed.
    """
    if _CUSTOM_EQ_MARKER in config_cls.__dict__:
        return
    if config_cls.__dict__.get("__eq__") is _canonical_eq:
        return
    config_cls.__eq__ = _canonical_eq  # type: ignore[method-assign]
    if config_cls.__dict__.get("__hash__", _MISSING) is None:
        config_cls.__hash__ = object.__hash__  # type: ignore[method-assign]


def config_equal(left: Any, right: Any) -> bool:
    """Compare two config objects by canonical value equality.

    The two objects are equal exactly when they are the same class and their
    compared fields (``init=True`` and ``compare=True``) serialize to the same
    canonical plain form, array fields compared through the backends' vectorized
    operations. Works on any config dataclass instance, ahead of any other
    engine call and whether or not the class subclasses ``ConfigRoot``, and
    touches no classes. The canonical relation is evaluated directly,
    independently of a custom root ``__eq__`` preserved by the class-body rule.

    Args:
        left: A config dataclass instance.
        right: The object to compare against.

    Returns:
        Whether the two objects are canonically equal.

    Raises:
        TypeError: When ``left`` is anything other than a dataclass instance.
    """
    if not is_dataclass(left) or isinstance(left, type):
        raise TypeError(f"config_equal() expects a config dataclass instance, got {type(left).__name__}")
    return _canonical_eq(left, right) is True
