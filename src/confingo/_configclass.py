"""Public ``@configclass`` decorator installing canonical equality on config schemas.

``configclass`` is a thin wrapper around ``dataclasses.dataclass``: it forwards
every dataclass keyword, generates the same fields and ``__init__``, and installs
an ``__eq__`` that compares two configs by their canonical serialized forms
(``to_dict(self) == to_dict(other)``). Canonical equality holds for every
supported field type, including array-valued fields whose backend ``==`` returns
elementwise results, so ``from_dict(cls, to_dict(config)) == config`` reads
literally on a decorated tree. Each decorated class is stamped with a marker
attribute that the engine checks during schema processing; a schema dataclass
lacking the marker triggers one ``ConfigWarning`` per class per process.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    dataclass_transform,
    overload,
)

from confingo._core import (
    _CONFIGCLASS_MARKER,
    to_dict,
)


if TYPE_CHECKING:
    import types
    from collections.abc import Callable

_T = TypeVar("_T")


class ConfigWarning(UserWarning):
    """Warning category for confingo schema advisories.

    Emitted once per class per process when a schema dataclass lacks the
    ``@configclass`` marker. Target it with ``warnings.filterwarnings`` to
    silence or escalate confingo advisories precisely.
    """


def _canonical_eq(self: Any, other: Any) -> bool | types.NotImplementedType:
    """Compare two config objects by their canonical serialized forms.

    Args:
        self: The left-hand config object.
        other: The right-hand operand.

    Returns:
        ``NotImplemented`` when ``other`` is a different class, else whether the
        two canonical ``to_dict`` forms are equal.
    """
    if other.__class__ is not self.__class__:
        return NotImplemented
    return to_dict(self) == to_dict(other)


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
    ``__init__`` are generated exactly as ``@dataclass`` generates them, and every
    dataclass keyword (``frozen``, ``kw_only``, ``slots``, ``order``, ...) is
    forwarded. ``eq`` is fixed to ``False`` so the decorator's own ``__eq__``
    is the single equality mechanism: it returns ``NotImplemented`` for a
    different class and otherwise compares ``to_dict(self)`` with
    ``to_dict(other)``. A user-defined ``__eq__`` in the class body is respected
    and left untouched. ``__hash__`` stays object identity; use ``config_hash``
    for value identity.

    Both the bare form (``@configclass``) and the parenthesized form
    (``@configclass(frozen=True)``) work, on root and section classes alike.

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
