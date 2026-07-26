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

Ordinary ``@dataclass`` declarations are the schema surface. A ``ConfigNode``
subclass carries canonical ``__eq__`` from class-creation time (installed by
``ConfigNode.__init_subclass__`` ahead of the ``@dataclass`` decorator, which
then skips generating its own), and every other schema dataclass receives it
at its first schema processing through ``_install_canonical_eq``. The
``config_equal`` free function exposes the same relation without touching the
classes involved.

confingo owns equality and hashing on config dataclasses: ``_install_canonical_eq``
rejects a class that hand-writes ``__eq__`` or ``__hash__``, or that declares a
``@dataclass`` flag it cannot honor (``init=False``, ``unsafe_hash=True``,
``eq=False``, ``order=True``), and sets ``__hash__`` to None so every config shares
one value-equality plus unhashable model and ``config_hash`` is the single
value-identity operation. A ``ConfigNode`` subclass carries the private
``_unhashable_config`` sentinel between class creation and that first touch, so a
node raises a remedy-naming ``TypeError`` for the whole window. Provenance is told
from a hand-written method by matching its code object against dataclass codegen on
the current interpreter; a method fabricated to be byte-identical to that codegen is
treated as generated.
"""

from __future__ import annotations

import math
from dataclasses import (
    fields,
    is_dataclass,
    make_dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

from confingo import _arrays
from confingo._errors import (
    ConfigError,
    ConfigIssue,
)
from confingo._schema import _classify_dataclass_fields
from confingo._serialize import (
    _canonical_json,
    _PlainProjection,
    _project_plain,
)


if TYPE_CHECKING:
    import types
    from collections.abc import Callable

_MISSING = object()
"""Sentinel distinguishing an absent class-dict entry from a stored None."""

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
      a (Any): The left-hand value.
      b (Any): The right-hand value.

    Returns:
      bool: Whether the two values serialize to equal plain forms.
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
      other (Any): The right-hand operand.

    Returns:
      bool | types.NotImplementedType: ``NotImplemented`` when ``other`` is a
        different class, else whether every compared field pair is canonically
        equal.
    """
    if other.__class__ is not self.__class__:
        return NotImplemented
    compared = _classify_dataclass_fields(self.__class__).compared
    return all(_values_equal(getattr(self, c.definition.name), getattr(other, c.definition.name)) for c in compared)


def _unhashable_config(self: Any) -> NoReturn:
    """Reject hashing a config object, naming the value-identity operation instead.

    Planted on every ``ConfigNode`` subclass at class creation, where a non-None
    ``__hash__`` is what makes ``@dataclass`` treat hashing as already decided and
    leave it alone. That covers the window before confingo first processes the
    class, after which ``__hash__`` becomes None and Python raises its own
    ``TypeError``.

    Args:
      self (Any): The config object a hash was requested for.

    Raises:
      TypeError: Always, naming ``config_hash`` as the value-identity operation.
    """
    raise TypeError(f"unhashable type: {type(self).__name__!r}; use config_hash(config) for value identity")


def custom_eq_message(name: str) -> str:
    """Build the rejection message for a config dataclass defining its own ``__eq__``.

    Args:
      name (str): The name of the offending class.

    Returns:
      str: The rejection message naming the class and the required remedy.
    """
    return (
        f"{name} defines a custom __eq__; confingo installs canonical value equality on config "
        f"dataclasses. Remove the __eq__ method so confingo owns equality."
    )


def custom_hash_message(name: str) -> str:
    """Build the rejection message for a config dataclass defining its own ``__hash__``.

    Args:
      name (str): The name of the offending class.

    Returns:
      str: The rejection message naming the class and the required remedy.
    """
    return (
        f"{name} defines a custom __hash__; confingo owns hashing and makes config dataclasses unhashable. "
        f"Remove the __hash__ method and use config_hash(config) for value identity."
    )


def _resolve_owner(config_cls: type[Any], name: str) -> tuple[type[Any], Any]:
    """Find the MRO class whose own dict provides ``name``, and that value.

    Resolving through the MRO rather than reading only ``config_cls.__dict__``
    catches an undecorated dataclass subclass that inherits a base's method: the
    subclass has an empty own dict yet ``is_dataclass`` still reports it.

    Args:
      config_cls (type[Any]): The class whose method to resolve.
      name (str): The dunder name to look up (``"__eq__"`` or ``"__hash__"``).

    Returns:
      tuple[type[Any], Any]: The owning class and the method object; the owner
        is ``object`` when nothing nearer defines it.
    """
    for base in config_cls.__mro__:
        if name in base.__dict__:
            return base, base.__dict__[name]
    return object, getattr(config_cls, name)


def _reference_code(owner: type[Any], names: list[str], *, frozen: bool) -> Any:
    """Build a reference method's code object from dataclass codegen for ``names``.

    ``make_dataclass`` runs the same equality / hash generator as ``@dataclass``
    on the current interpreter, so the reference carries the version-exact code
    object confingo compares provenance against.

    Args:
      owner (type[Any]): The class whose provenance is being checked, used only
        for the reference name.
      names (list[str]): Field names, in declaration order, the generated method
        ranges over.
      frozen (bool): Whether to build a frozen reference, which generates
        ``__hash__``; a mutable reference generates only ``__eq__``.

    Returns:
      Any: The generated ``__eq__`` (mutable) or ``__hash__`` (frozen) code object.
    """
    reference = make_dataclass(f"_ConfingoRef_{owner.__name__}", [(name, int) for name in names], frozen=frozen)
    method = reference.__hash__ if frozen else reference.__eq__
    return method.__code__


def _code_matches(method: Any, reference_code: Any) -> bool:
    """Report whether a method's code object matches a dataclass-generated reference.

    Matching ``co_code`` plus ``co_consts``, ``co_names``, and ``co_filename`` is
    the strictest practical detector. A method deliberately fabricated (e.g. via
    ``exec``) to be byte-identical to dataclass codegen is treated as generated;
    confingo does not defend against that.

    Args:
      method (Any): The method object under inspection.
      reference_code (Any): The generated reference code object.

    Returns:
      bool: Whether the method's code object is byte-identical to the reference.
    """
    code = getattr(method, "__code__", None)
    if code is None:
        return False
    return (
        code.co_code == reference_code.co_code
        and code.co_consts == reference_code.co_consts
        and code.co_names == reference_code.co_names
        and code.co_filename == reference_code.co_filename
    )


def _custom_method_owner(
    config_cls: type[Any],
    name: str,
    *,
    owned: Callable[[Any], bool],
    names_of: Callable[[type[Any]], list[str]],
    frozen: bool,
) -> type[Any] | None:
    """Report the class owning a hand-written dunder, or None when confingo owns it.

    Resolves the method through the MRO, treats the owned sentinels and any
    dataclass-generated body (matched by provenance against ``names_of`` codegen)
    as owned, and otherwise names the class that hand-wrote it.

    Args:
      config_cls (type[Any]): The class to inspect.
      name (str): The dunder to resolve (``"__eq__"`` or ``"__hash__"``).
      owned (Callable[[Any], bool]): Whether a resolved method is a confingo- or
        object-owned sentinel that needs no provenance check.
      names_of (Callable[[type[Any]], list[str]]): The field names the generated
        reference method ranges over, in declaration order.
      frozen (bool): Whether the reference method is generated frozen, selecting
        ``__hash__`` codegen over ``__eq__`` codegen.

    Returns:
      type[Any] | None: The owning class when the method is hand-written, else
        None when confingo, ``object``, or dataclass codegen owns it.
    """
    owner, method = _resolve_owner(config_cls, name)
    if owned(method):
        return None
    if is_dataclass(owner) and _code_matches(method, _reference_code(owner, names_of(owner), frozen=frozen)):
        return None
    return owner


def _custom_eq_owner(config_cls: type[Any]) -> type[Any] | None:
    """Report the class owning a custom ``__eq__``, or None when equality is owned.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      type[Any] | None: The owning class when its ``__eq__`` is hand-written,
        else None for ``_canonical_eq``, ``object.__eq__``, or dataclass-generated.
    """
    return _custom_method_owner(
        config_cls,
        "__eq__",
        owned=lambda method: method is _canonical_eq or method is object.__eq__ or method is None,
        names_of=lambda owner: [field.name for field in fields(owner) if field.compare],
        frozen=False,
    )


def _custom_hash_owner(config_cls: type[Any]) -> type[Any] | None:
    """Report the class owning a custom ``__hash__``, or None when hashing is owned.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      type[Any] | None: The owning class when its ``__hash__`` is hand-written,
        else None for ``object.__hash__``, an unhashable ``None``, confingo's own
        ``_unhashable_config`` sentinel, or the dataclass-generated frozen hash.
    """
    return _custom_method_owner(
        config_cls,
        "__hash__",
        owned=lambda method: method is None or method is object.__hash__ or method is _unhashable_config,
        names_of=lambda owner: [
            field.name for field in fields(owner) if (field.compare if field.hash is None else field.hash)
        ],
        frozen=True,
    )


def _flag_conflicts(config_cls: type[Any]) -> list[str]:
    """Collect messages for ``@dataclass`` flags confingo cannot honor.

    Params are resolved through the MRO (``getattr``) so an undecorated dataclass
    subclass is governed by the base's decoration; at schema processing decoration
    is complete, so inherited params are authoritative.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      list[str]: One message per violated flag, empty when the flags are honored.
    """
    params = getattr(config_cls, "__dataclass_params__", None)
    if params is None:
        return []
    name = config_cls.__name__
    messages: list[str] = []
    if params.init is False:
        messages.append(
            f"{name} is declared @dataclass(init=False); confingo builds config objects by calling the "
            f"class, which needs the generated __init__. Use init=True (the default)."
        )
    if getattr(params, "unsafe_hash", False) is True:
        messages.append(
            f"{name} is declared @dataclass(unsafe_hash=True); confingo owns hashing and makes config "
            f"dataclasses unhashable. Use the default hashing and config_hash(config) for value identity."
        )
    if params.eq is False:
        messages.append(
            f"{name} is declared @dataclass(eq=False); confingo owns equality and installs canonical __eq__. "
            f"Use eq=True (the default)."
        )
    if params.order is True:
        messages.append(
            f"{name} is declared @dataclass(order=True); ordering compares the raw field tuple, which "
            f"disagrees with canonical equality and raises on array fields. Use the default ordering."
        )
    return messages


def _enforce_class_contract(config_cls: type[Any]) -> None:
    """Reject a config dataclass that confingo cannot own equality and hashing for.

    Gathers every violation -- conflicting ``@dataclass`` flags, a hand-written
    ``__eq__``, and a hand-written ``__hash__`` -- into one error so a class with
    several problems reports them together. Runs before any attribute mutation so
    a rejected class is never partially modified.

    Args:
      config_cls (type[Any]): The schema dataclass being processed.

    Raises:
      ConfigError: When the class carries any contract violation.
    """
    issues = [ConfigIssue(path="", message=message) for message in _flag_conflicts(config_cls)]
    eq_owner = _custom_eq_owner(config_cls)
    if eq_owner is not None:
        issues.append(ConfigIssue(path="", message=custom_eq_message(eq_owner.__name__)))
    hash_owner = _custom_hash_owner(config_cls)
    if hash_owner is not None:
        issues.append(ConfigIssue(path="", message=custom_hash_message(hash_owner.__name__)))
    if len(issues) > 0:
        raise ConfigError(issues, context="config schema")


def _disable_hash(config_cls: type[Any]) -> None:
    """Make a config dataclass unhashable by setting its own ``__hash__`` to None.

    Whatever the class arrives with -- ``object.__hash__``, the None of a mutable
    dataclass, the field-tuple hash of a frozen one, the ``_unhashable_config``
    sentinel a node carries from class creation, or any of those inherited by an
    undecorated dataclass subclass -- ``config_cls`` gets its own ``None`` entry.
    Writing it on the touched class shadows an inherited hash rather than mutating
    the base, and after the write ``hash(config)`` raises Python's own
    ``TypeError`` while ``config_hash`` carries value identity. A hand-written hash
    is rejected earlier, so only confingo-owned and generated hashes reach here.

    Args:
      config_cls (type[Any]): The schema dataclass being processed.
    """
    if config_cls.__dict__.get("__hash__", _MISSING) is None:
        return
    config_cls.__hash__ = None  # type: ignore[method-assign]


def _install_canonical_eq(config_cls: type[Any]) -> None:
    """Install canonical equality and the unhashable contract on a schema dataclass.

    The class is first checked against confingo's ownership contract (no custom
    ``__eq__`` / ``__hash__``, no conflicting ``@dataclass`` flags); only then is
    ``_canonical_eq`` installed in place of the generated ``__eq__`` and
    ``__hash__`` set to None, so every config dataclass shares one equality
    contract however it was declared and ``config_hash`` is the one value-identity
    operation. A ``ConfigNode`` subclass already carries ``_canonical_eq`` from
    class-creation time, and its sentinel hash is replaced here.

    Args:
      config_cls (type[Any]): The schema dataclass being processed.

    Raises:
      ConfigError: When the class violates the ownership contract.
    """
    _enforce_class_contract(config_cls)
    if config_cls.__dict__.get("__eq__") is not _canonical_eq:
        config_cls.__eq__ = _canonical_eq  # type: ignore[method-assign]
    _disable_hash(config_cls)


def config_equal(left: Any, right: Any) -> bool:
    """Compare two config objects by canonical value equality.

    The two objects are equal exactly when they are the same class and their
    compared fields (``init=True`` and ``compare=True``) serialize to the same
    canonical plain form, array fields compared through the backends' vectorized
    operations. Works on any config dataclass instance, ahead of any other
    engine call and whether or not the class subclasses ``ConfigNode``, and
    touches no classes. The canonical relation is evaluated directly, on the
    two objects as given.

    Args:
      left (Any): A config dataclass instance.
      right (Any): The object to compare against.

    Returns:
      bool: Whether the two objects are canonically equal.

    Raises:
      TypeError: When ``left`` is anything other than a dataclass instance.
    """
    if not is_dataclass(left) or isinstance(left, type):
        raise TypeError(f"config_equal() expects a config dataclass instance, got {type(left).__name__}")
    return _canonical_eq(left, right) is True
