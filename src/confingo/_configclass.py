"""Public ``@configclass`` decorator and confingo's canonical equality engine.

``configclass`` is a thin wrapper around ``dataclasses.dataclass``: it forwards
the layout keywords, generates the same fields and ``__init__``, and installs a
canonical ``__eq__`` under which two configs are equal exactly when they
serialize to the same plain form. Array and tensor fields compare through the
backends' vectorized operations wherever that comparison is provably exact,
dataclass sections and containers recurse structurally, and every remaining
value pair compares by its ``to_dict`` form, so ``==`` runs at native speed on
large arrays while ``from_dict(cls, to_dict(config)) == config`` reads
literally on a decorated tree.

The same canonical ``__eq__`` installs onto plain ``@dataclass`` schema classes
at their first schema processing (see ``_install_canonical_eq``), so sections
declared with either decorator share one equality contract.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
    fields,
    is_dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    dataclass_transform,
    overload,
)

from confingo import _arrays
from confingo._core import (
    _CONFIGCLASS_MARKER,
    to_dict,
)


if TYPE_CHECKING:
    import types
    from collections.abc import Callable

_T = TypeVar("_T")

_MISSING = object()
"""Sentinel distinguishing an absent class-dict entry from a stored None."""

_EXACT_PRIMITIVES = (bool, int, float, str)
"""Builtin scalar types whose own ``==`` matches plain-form comparison exactly.

Membership is by exact type: subclasses such as ``np.float64`` (whose ``==``
applies NumPy promotion) and enum members (whose ``==`` may be overridden)
compare through their serialized forms instead.
"""


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two field values by their canonical serialized forms.

    Array pairs of the loaded backends compare through
    ``_arrays.native_equal`` where its vectorized path applies. Exact-type
    builtin scalars compare directly, dataclass pairs of the same class,
    sequence pairs, and str-keyed dict pairs recurse structurally, and
    every other pair -- scalar subclasses, enum members, dicts with
    canonicalizing keys -- compares by its ``to_dict`` form. Each branch
    agrees with plain-form comparison on the supported value domain, so the
    walk is an evaluation strategy for one equality relation.

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
    if type(a) in _EXACT_PRIMITIVES and type(b) in _EXACT_PRIMITIVES:
        return bool(a == b)
    if is_dataclass(a) and not isinstance(a, type) and a.__class__ is b.__class__:
        return all(_values_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
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
    return bool(to_dict(a) == to_dict(b))


def _canonical_eq(self: Any, other: Any) -> bool | types.NotImplementedType:
    """Compare two config objects by canonical value equality.

    Equality means the two objects serialize to the same canonical plain
    form: every field compares through ``_values_equal``, which runs array
    fields through the backends' vectorized comparisons and recurses through
    sections and containers.

    Args:
        self: The left-hand config object.
        other: The right-hand operand.

    Returns:
        ``NotImplemented`` when ``other`` is a different class, else whether
        every field pair is canonically equal.
    """
    if other.__class__ is not self.__class__:
        return NotImplemented
    return all(_values_equal(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))


def _install_canonical_eq(config_cls: type[Any]) -> None:
    """Install canonical equality on a schema dataclass at schema processing.

    The ``@configclass`` marker is the single signal: a marked class already
    carries its equality contract from decoration time (a body-defined
    ``__eq__`` included) and is left alone, and every unmarked schema
    dataclass has ``_canonical_eq`` installed in place of the ``__eq__`` it
    carried, with a ``__hash__`` slot the dataclass set to None reverting to
    identity hashing. A schema class that needs a custom ``__eq__`` declares
    it in a ``@configclass`` body, where the decorator respects it.

    Args:
        config_cls: The schema dataclass being processed.
    """
    if _CONFIGCLASS_MARKER in config_cls.__dict__:
        return
    if config_cls.__dict__.get("__eq__") is _canonical_eq:
        return
    config_cls.__eq__ = _canonical_eq  # type: ignore[method-assign]
    if config_cls.__dict__.get("__hash__", _MISSING) is None:
        config_cls.__hash__ = object.__hash__  # type: ignore[method-assign]


@overload
def configclass(cls: type[_T], /) -> type[_T]: ...


@overload
def configclass(
    *,
    init: bool = True,
    repr: bool = True,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> Callable[[type[_T]], type[_T]]: ...


@dataclass_transform(field_specifiers=(field,))
def configclass(cls: type[_T] | None = None, /, **kwargs: Any) -> type[_T] | Callable[[type[_T]], type[_T]]:
    """Declare a config dataclass with canonical equality.

    A thin wrapper around ``dataclasses.dataclass``: fields, defaults, and
    ``__init__`` are generated exactly as ``@dataclass`` generates them, and
    the layout keywords (``frozen``, ``kw_only``, ``slots``, ...) forward.
    ``eq`` is fixed to ``False`` so the decorator's own ``__eq__`` is the
    single equality mechanism: it returns ``NotImplemented`` for a different
    class and otherwise compares the two objects' canonical serialized
    values, with array fields running through the backends' vectorized
    comparisons. A user-defined ``__eq__`` in the class body is respected and
    left untouched. ``__hash__`` stays object identity; use ``config_hash``
    for value identity.

    Both the bare form (``@configclass``) and the parenthesized form
    (``@configclass(frozen=True)``) work, on root and section classes alike.
    Plain ``@dataclass`` schema classes receive the same canonical ``__eq__``
    at their first schema processing; decorating is the way to carry it from
    class-creation time and to state the schema role explicitly.

    Args:
        cls: The class being decorated in the bare form, positional-only.
        **kwargs: Keyword arguments forwarded to ``dataclasses.dataclass``.

    Returns:
        The decorated class in the bare form, or a class decorator in the
        parenthesized form.

    Raises:
        TypeError: When ``eq`` is passed explicitly; the decorator controls
          ``__eq__`` and fixes ``eq=False``. Also when ``order`` or
          ``unsafe_hash`` is true: ordering builds on the generated ``__eq__``
          that ``configclass`` replaces, and a generated field hash would
          contradict the identity-``__hash__`` contract and fail on
          array-valued fields.
    """
    if "eq" in kwargs:
        raise TypeError("configclass() installs canonical __eq__ itself and fixes eq=False; drop the eq argument")
    if kwargs.get("order") is True:
        raise TypeError("configclass() replaces the generated __eq__ that dataclass ordering builds on; drop order")
    if kwargs.get("unsafe_hash") is True:
        raise TypeError("configclass() keeps __hash__ as object identity; drop unsafe_hash and use config_hash")

    def wrap(inner: type[_T]) -> type[_T]:
        built = dataclass(eq=False, **kwargs)(inner)
        if "__eq__" not in built.__dict__:
            built.__eq__ = _canonical_eq  # type: ignore[method-assign]
        setattr(built, _CONFIGCLASS_MARKER, True)
        return built

    if cls is None:
        return wrap
    return wrap(cls)
