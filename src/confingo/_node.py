"""Base class exposing the marshal / unmarshal helpers under one accessor.

A config dataclass that subclasses ``ConfigNode`` gains the free-function
helpers under ``cfg``: the ``from_*`` / ``load_*`` builders and ``validate_schema``
answer on the class and on a value alike, and the ``to_*`` / ``dumps_*`` /
``save_*`` / ``hash`` operations read the value they are called on, each
delegating to the matching free function. Any dataclass in a config tree may
subclass it, at any depth; a section that stays a plain dataclass is walked by
introspection exactly as before, and the free functions cover it.

Each operation is scoped to the node it is reached through:
``node.cfg.to_dict()`` renders that node's subtree, ``node.cfg.hash()``
fingerprints that subtree, and ``Node.cfg.from_dict(...)`` reports issue paths
relative to that node. The engine reaches a nested node through the same generic
recursion it uses for a plain dataclass, so subclassing adds the accessor while
the tree builds, serializes, and validates the same way.
"""

from __future__ import annotations

from dataclasses import (
    fields,
    is_dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    overload,
)

from confingo._core import from_dict as _from_dict
from confingo._core import validate_schema as _validate_schema
from confingo._equality import (
    _canonical_eq,
    _custom_eq_owner,
    _custom_hash_owner,
    _unhashable_config,
    custom_eq_message,
    custom_hash_message,
)
from confingo._errors import ConfigError as _ConfigError
from confingo._errors import ConfigIssue as _ConfigIssue
from confingo._errors import class_label as _class_label
from confingo._file import from_file as _from_file
from confingo._file import to_file as _to_file
from confingo._json import dumps_json as _dumps_json
from confingo._json import load_json as _load_json
from confingo._json import save_json as _save_json
from confingo._schema import own_annotations as _own_annotations
from confingo._serialize import config_hash as _config_hash
from confingo._serialize import to_dict as _to_dict
from confingo._yaml import dumps_yaml as _dumps_yaml
from confingo._yaml import load_yaml as _load_yaml
from confingo._yaml import save_yaml as _save_yaml


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def facade_collision_message(class_name: str, attr_name: str, source: str) -> str:
    """Build the rejection message for a name that shadows a ``ConfigNode`` method.

    Args:
      class_name (str): The name of the offending class.
      attr_name (str): The reserved method name that is shadowed.
      source (str): Where the shadowing member comes from, phrased to complete
        the sentence (``"is declared as a field"``).

    Returns:
      str: The rejection message naming the class, the reserved name, and the
        required remedy.
    """
    return (
        f"{class_name}.{attr_name} {source}, which shadows the ConfigNode method of the same name; "
        f"rename it so the node keeps its {attr_name} method."
    )


def _facade_collisions(cls: type[Any]) -> list[str]:
    """Collect messages for reserved method names a subclass shadows.

    Every check reads declarations rather than resolving them. Resolving would
    run whatever descriptor the author supplied, which can carry side effects at
    class-creation time, and its class-access result does not establish how the
    name behaves on an instance: a descriptor can hand back the real method for
    class access and something else for instance access. A declaration that
    outranks the facade is enough to reject, so the checks stay static.

    Args:
      cls (type[Any]): The subclass being created.

    Returns:
      list[str]: One message per shadowed name, in sorted name order, empty when
        the facade is intact.
    """
    messages: list[str] = []
    for name in _FACADE_NAMES:
        source = _collision_source(cls, name)
        if source is not None:
            messages.append(facade_collision_message(_class_label(cls), name, source))
    return messages


def _collision_source(cls: type[Any], name: str) -> str | None:
    """Describe how ``cls`` loses the facade method ``name``, or None when intact.

    A subclass may not declare a reserved name at all, in any form: the
    annotation alone is rejected without asking whether ``@dataclass`` would
    store it, since deciding that from a string annotation would mean
    reproducing the standard library's alias resolution, and a node has no
    reason to spell one of its own method names. What the class inherits is
    judged by what actually lands on it instead, which ``dataclasses.fields``
    answers exactly.

    Args:
      cls (type[Any]): The subclass being created.
      name (str): The reserved method name to check.

    Returns:
      str | None: A phrase completing "``Class.name`` ...", else None.
    """
    if name in _own_annotations(cls):
        return "is declared as a field"
    if name in cls.__dict__:
        return "is bound in the class body"
    field_owner = _inherited_field_owner(cls, name)
    if field_owner is not None:
        return f"is inherited as a field from {_class_label(field_owner)}"
    binding_owner = _preceding_binding_owner(cls, name)
    if binding_owner is not None:
        return f"is supplied by base {_class_label(binding_owner)}"
    metaclass_owner = _metaclass_shadow_owner(cls, name)
    if metaclass_owner is not None:
        return f"is supplied as a data descriptor by metaclass {_class_label(metaclass_owner)}"
    return None


def _preceding_binding_owner(cls: type[Any], name: str) -> type[Any] | None:
    """Find a base that binds ``name`` ahead of ``ConfigNode`` in the MRO.

    Attribute lookup stops at the first class whose dict holds the name, so any
    binding ahead of ``ConfigNode`` takes the facade's place, whatever kind of
    object it is. A binding after ``ConfigNode`` loses to the method and is left
    alone.

    Args:
      cls (type[Any]): The subclass being created.
      name (str): The reserved method name to look for.

    Returns:
      type[Any] | None: The binding base, or None when the facade resolves.
    """
    for base in cls.__mro__[1:]:
        if base is ConfigNode:
            return None
        if name in base.__dict__:
            return base
    return None


def _metaclass_shadow_owner(cls: type[Any], name: str) -> type[Any] | None:
    """Find a metaclass supplying ``name`` as a data descriptor.

    Class-attribute lookup gives a data descriptor on the metaclass precedence
    over the class's own MRO, so one named after a builder classmethod replaces
    it on the class. Both the binding and its descriptor kind are read from raw
    class namespaces rather than by resolving anything, so neither the descriptor
    nor any lookup hook around it runs. A non-data metaclass binding loses to the
    MRO and is left alone.

    Args:
      cls (type[Any]): The subclass being created.
      name (str): The reserved method name to look for.

    Returns:
      type[Any] | None: The metaclass holding the data descriptor, else None.
    """
    for meta in type(cls).__mro__:
        if name not in meta.__dict__:
            continue
        if _is_data_descriptor(meta.__dict__[name]):
            return meta
        return None
    return None


def _is_data_descriptor(value: Any) -> bool:
    """Report whether ``value`` implements the data-descriptor protocol.

    Membership is read from the raw type namespaces along the MRO through
    ``type.__getattribute__``, which bypasses any ``__getattribute__`` the
    descriptor's own metaclass defines. An ordinary ``hasattr`` would run that
    hook, which can execute code or deny a slot the protocol still honors.

    Args:
      value (Any): The object bound in a metaclass namespace.

    Returns:
      bool: True when the type defines ``__set__`` or ``__delete__``.
    """
    descriptor_type = type(value)
    for klass in type.__getattribute__(descriptor_type, "__mro__"):
        namespace = type.__getattribute__(klass, "__dict__")
        if "__set__" in namespace or "__delete__" in namespace:
            return True
    return False


def _inherited_field_owner(cls: type[Any], name: str) -> type[Any] | None:
    """Find the base that first declares ``name`` as a stored dataclass field.

    Every dataclass base is searched regardless of its position relative to
    ``ConfigNode``, since a stored field reaches the instance through the
    generated ``__init__`` and shadows the method from either side of the MRO.

    ``__dataclass_fields__`` also carries ``ClassVar`` and ``InitVar``
    pseudo-fields, which the generated ``__init__`` never stores on the
    instance, so membership alone would reject safe schemas. Each base's entry
    is checked against ``dataclasses.fields``, which lists stored fields only.

    The base-ward walk mirrors what ``@dataclass`` itself does when it collects
    inherited fields: metadata is read through ordinary attribute lookup, so an
    undecorated subclass of a dataclass contributes its base's fields rather
    than being skipped. That matters in a diamond, where such a base can
    reintroduce a stored field after another branch replaced it with a
    ``ClassVar``. The most-derived contribution decides.

    Attribution is by ``Field`` identity rather than position, since the
    decorator shares one ``Field`` object across every class that inherits it.
    That names the class whose own metadata introduced the surviving field.

    Args:
      cls (type[Any]): The subclass being created.
      name (str): The reserved method name to look for.

    Returns:
      type[Any] | None: The introducing base when the effective declaration is a
        stored field, else None.
    """
    effective: Any = None
    for base in reversed(cls.__mro__[1:]):
        declared = getattr(base, "__dataclass_fields__", None)
        if declared is None or name not in declared:
            continue
        effective = declared[name] if any(field.name == name for field in fields(base)) else None
    if effective is None:
        return None
    return next(
        (
            base
            for base in reversed(cls.__mro__[1:])
            if base.__dict__.get("__dataclass_fields__", {}).get(name) is effective
        ),
        None,
    )


class _ConfigFacade[NodeT]:
    """The operations a config class answers, bound to the class it was reached through.

    Reached as ``Config.cfg``. The builders and ``validate_schema`` read the class, so
    a class is all they need. ``_ValueFacade`` extends this with the operations
    that read a config object, and is what instance access hands out.
    """

    __slots__ = ("_instance", "_owner")

    def __init__(self, owner: type[NodeT], instance: NodeT | None) -> None:
        """Bind the facade to one class and, when reached from a value, that value.

        Args:
          owner (type[NodeT]): The config class the facade was reached through.
          instance (NodeT | None): The config object it was reached through, or
            None for class access.
        """
        self._owner = owner
        self._instance = instance

    def _value(self, operation: str) -> NodeT:
        """Return the config object this facade is bound to.

        Args:
          operation (str): The operation being called, named in the message.

        Returns:
          NodeT: The bound config object.

        Raises:
          TypeError: When the facade was reached through the class, which carries
            no value to operate on.
        """
        if self._instance is None:
            raise TypeError(
                f"{_class_label(self._owner)}.cfg.{operation}() reads a config object; "
                f"call it on an instance, as config.cfg.{operation}()"
            )
        return self._instance

    def from_dict(self, data: Mapping[str, Any], *, context: str = "config") -> NodeT:
        """Build an instance from a nested mapping, reporting every problem at once.

        Issue paths are relative to this node, so a leaf that reports as
        ``trainer.lr`` when built through the enclosing config reports as ``lr``
        here.

        Args:
          data (Mapping[str, Any]): Nested mapping of config values, typically
            parsed from a config file.
          context (str = "config"): Description of the config source used in the
            error summary.

        Returns:
          NodeT: The constructed config object, typed as the class the facade
            was reached through.

        Raises:
          ConfigError: When the mapping fails to build; the exception lists every
            issue found.
        """
        return _from_dict(self._owner, data, context=context)

    def load_json(self, path: str | Path) -> NodeT:
        """Load a JSON file into an instance.

        Args:
          path (str | Path): Path to the JSON file.

        Returns:
          NodeT: The constructed config object, typed as the class the facade
            was reached through.

        Raises:
          ConfigError: When the file is unreadable, malformed, non-mapping, or
            fails validation.
        """
        return _load_json(self._owner, path)

    def load_yaml(self, path: str | Path) -> NodeT:
        """Load a YAML file into an instance.

        Args:
          path (str | Path): Path to the YAML file.

        Returns:
          NodeT: The constructed config object, typed as the class the facade
            was reached through.

        Raises:
          ConfigError: When the file is unreadable, malformed, non-mapping, or
            fails validation.
        """
        return _load_yaml(self._owner, path)

    def from_file(self, path: str | Path) -> NodeT:
        """Load a config file into an instance, choosing the reader by extension.

        A ``.json`` path reads JSON; a ``.yaml`` or ``.yml`` path reads YAML.

        Args:
          path (str | Path): Path to the config file.

        Returns:
          NodeT: The constructed config object, typed as the class the facade
            was reached through.

        Raises:
          ConfigError: When the extension names no supported format, or the file
            is unreadable, malformed, non-mapping, or fails validation.
        """
        return _from_file(self._owner, path)

    def validate_schema(self, *, context: str = "config schema") -> None:
        """Check this class's schema without building anything from it.

        Walks the whole tree the class declares, recursing into nested sections
        and into sections held in lists, tuples, sets, and dict values. No config
        data is read and no ``default_factory`` is called.

        Args:
          context (str = "config schema"): Description of the schema used in the
            error summary.

        Raises:
          ConfigError: When the schema carries any issue; the exception lists
            every issue found in the whole tree.
        """
        _validate_schema(self._owner, context=context)


class _ValueFacade[NodeT](_ConfigFacade[NodeT]):
    """Every config operation, bound to the config object it was reached through.

    Reached as ``config.cfg``. It carries the class operations of
    ``_ConfigFacade`` along with the operations that render, write, or
    fingerprint the bound value.
    """

    __slots__ = ()

    def to_dict(self) -> Any:
        """Convert this node's subtree into plain JSON-safe Python data.

        Returns:
          Any: The converted plain-data structure.
        """
        return _to_dict(self._value("to_dict"))

    def dumps_json(self, *, indent: int = 2) -> str:
        """Render this node's subtree as JSON text.

        Args:
          indent (int = 2): Number of spaces per indentation level.

        Returns:
          str: The JSON document, ending with a newline.
        """
        return _dumps_json(self._value("dumps_json"), indent=indent)

    def save_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this node's subtree to a JSON file, replacing the target atomically.

        The document holds this node's fields, so it loads back through this
        node's class.

        Args:
          path (str | Path): Destination file path. Parent directories are created as needed.
          indent (int = 2): Number of spaces per indentation level.

        Returns:
          Path: The path written.
        """
        return _save_json(self._value("save_json"), path, indent=indent)

    def dumps_yaml(self, *, indent: int = 2, sort_keys: bool = False) -> str:
        """Render this node's subtree as a YAML document.

        Args:
          indent (int = 2): Number of spaces per indentation level.
          sort_keys (bool = False): Whether to sort mapping keys.

        Returns:
          str: The YAML document, ending with a newline.
        """
        return _dumps_yaml(self._value("dumps_yaml"), indent=indent, sort_keys=sort_keys)

    def save_yaml(self, path: str | Path, *, indent: int = 2, sort_keys: bool = False) -> Path:
        """Write this node's subtree to a YAML file, replacing the target atomically.

        The document holds this node's fields, so it loads back through this
        node's class.

        Args:
          path (str | Path): Destination file path. Parent directories are created as needed.
          indent (int = 2): Number of spaces per indentation level.
          sort_keys (bool = False): Whether to sort mapping keys.

        Returns:
          Path: The path written.
        """
        return _save_yaml(self._value("save_yaml"), path, indent=indent, sort_keys=sort_keys)

    def to_file(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this node's subtree to a file, choosing the writer by extension.

        A ``.json`` path writes JSON; a ``.yaml`` or ``.yml`` path writes YAML.

        Args:
          path (str | Path): Destination file path. Parent directories are created as needed.
          indent (int = 2): Number of spaces per indentation level.

        Returns:
          Path: The path written.

        Raises:
          ConfigError: When the extension names no supported format.
        """
        return _to_file(self._value("to_file"), path, indent=indent)

    def hash(self, *, length: int = 12) -> str:
        """Fingerprint this node's subtree with a stable digest over its canonical JSON form.

        The digest ranges over the hashing fields (``init=True``, ``compare=True``,
        effective hash enabled), so a ``compare=False`` or ``hash=False`` field is
        carried by ``to_dict`` yet excluded from the digest. The digest covers
        this node's subtree, so a nested node fingerprints its own section rather
        than the enclosing config.

        Args:
          length (int = 12): Number of leading hex characters to return.

        Returns:
          str: The truncated SHA-256 digest.
        """
        return _config_hash(self._value("hash"), length=length)


class _CfgAccessor:
    """The descriptor that hands out a bound facade.

    One reserved name carries every operation, so a config class is free to name
    its own fields and methods anything else. Class access is typed as the class
    operations alone and value access as every operation, so an operation that
    reads a config object is offered where a config object exists.
    """

    @overload
    def __get__[NodeT](self, instance: None, owner: type[NodeT]) -> _ConfigFacade[NodeT]: ...

    @overload
    def __get__[NodeT](self, instance: NodeT, owner: type[NodeT] | None = None) -> _ValueFacade[NodeT]: ...

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> _ConfigFacade[Any]:
        """Bind the facade to whichever of the class and the instance was used.

        Class access carries the class it was reached through, and instance
        access carries the instance's own type, so a builder answers with the
        subclass the caller named rather than the base. One runtime type carries
        every operation, so a value operation reached from the class raises the
        naming ``TypeError`` that names the instance form.

        Args:
          instance (Any): The config object, or None for class access.
          owner (type[Any] | None = None): The class the attribute was reached through.

        Returns:
          _ConfigFacade[Any]: The ``_ValueFacade`` bound to that class and instance.
        """
        if owner is None:
            owner = type(instance)
        return _ValueFacade(owner, instance)

    def __set__(self, instance: Any, value: Any) -> NoReturn:
        """Reject assigning over the accessor, naming what the name carries.

        Defining this alongside ``__get__`` makes the accessor a data descriptor,
        which is what puts it ahead of an instance attribute of the same name. An
        instance-dict entry would otherwise win, and the node would silently stop
        answering the surface every other guard on this name exists to protect.

        Args:
          instance (Any): The config object being assigned to.
          value (Any): The value offered.

        Raises:
          AttributeError: Always, naming the accessor and the operations it carries.
        """
        raise AttributeError(
            f"cfg carries {_class_label(type(instance))}'s config operations and cannot be assigned; "
            f"call one of them, such as config.cfg.to_dict(), or name a field something other than cfg"
        )

    def __delete__(self, instance: Any) -> NoReturn:
        """Reject deleting the accessor, naming what the name carries.

        Args:
          instance (Any): The config object the deletion was aimed at.

        Raises:
          AttributeError: Always, naming the accessor and the operations it carries.
        """
        raise AttributeError(
            f"cfg carries {_class_label(type(instance))}'s config operations and cannot be deleted; "
            f"call one of them, such as config.cfg.to_dict()"
        )


class ConfigNode:
    """Mixin adding marshal / unmarshal methods to a config dataclass.

    Subclass this on any config dataclass, at any depth in the tree, then
    decorate it with ``@dataclass`` as usual. The class carries its own schema,
    so building and loading read as ``Config.cfg.load_json(path)`` rather than
    ``load_json(Config, path)``, and a nested section that subclasses it gains
    the same operations over its own subtree.

    Subclassing reserves one name, ``cfg``, which carries every operation: a
    field, class-body binding, or base-supplied member of that name is rejected
    at class creation, since it would resolve ahead of the accessor and leave the
    node's advertised surface unusable. Every other name a config class might
    want, ``validate_schema`` and ``to_dict`` among them, stays free.

    Subclassing also installs canonical equality from class-creation time:
    ``__init_subclass__`` plants the canonical ``__eq__`` and a raising
    ``__hash__`` into the subclass ahead of the ``@dataclass`` decorator, which
    then keeps them in place of generating its own. A subclass whose body
    defines its own ``__eq__`` or ``__hash__`` is rejected here, as is one that
    inherits a hand-written ``__eq__`` or ``__hash__`` from a base, since
    confingo owns equality and hashing on config dataclasses and the canonical
    methods would otherwise mask the inherited definition. A conflicting
    ``@dataclass`` flag is rejected later, at first schema processing, once
    decoration has run. Because a non-None ``__hash__`` lands ahead of the
    decorator, a subclass declared ``@dataclass(unsafe_hash=True)`` fails at
    class creation with the standard-library ``TypeError`` for overwriting
    ``__hash__``; a plain dataclass carrying that flag reports a ``ConfigError``
    instead.

    Config objects are unhashable. The planted ``__hash__`` raises a
    ``TypeError`` naming ``config_hash`` for the window between class creation
    and first schema processing, which is also what holds a frozen subclass to
    that contract; from first schema processing on, the class carries
    ``__hash__ = None`` like every other config dataclass.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reserve the facade names and install canonical equality and unhashability.

        Args:
          **kwargs (Any): Keyword arguments forwarded to ``super().__init_subclass__``.

        Raises:
          ConfigError: When the subclass shadows a reserved method name, defines
            its own ``__eq__`` or ``__hash__``, or inherits a hand-written
            ``__eq__`` or ``__hash__``; every violation is reported together.
        """
        super().__init_subclass__(**kwargs)
        messages = _facade_collisions(cls)

        # Both resolve from cls outward, so one call covers a definition in this
        # class's own body and one it would otherwise mask on a base.
        eq_owner = _custom_eq_owner(cls)
        if eq_owner is not None:
            messages.append(custom_eq_message(_class_label(eq_owner)))
        hash_owner = _custom_hash_owner(cls)
        if hash_owner is not None:
            messages.append(custom_hash_message(_class_label(hash_owner)))

        if len(messages) > 0:
            raise _ConfigError(
                [_ConfigIssue(path="", message=message) for message in messages], context="config schema"
            )
        if cls.__dict__.get("__eq__") is None:
            cls.__eq__ = _canonical_eq  # type: ignore[method-assign]
            # A non-None __hash__ is what dataclasses read as "hashing is already
            # decided", so this sentinel is what keeps @dataclass(frozen=True)
            # from generating a field-tuple hash over the canonical __eq__ landing
            # beside it. First schema touch replaces it with None.
            cls.__hash__ = _unhashable_config  # type: ignore[method-assign]

    cfg = _CfgAccessor()
    """Every config operation, reached through one name.

    ``Config.cfg`` carries the builders and the schema check; ``config.cfg``
    carries those plus the operations that read a value.
    """


_FACADE_NAMES: tuple[str, ...] = tuple(sorted(name for name in vars(ConfigNode) if not name.startswith("_")))
"""The names a ``ConfigNode`` subclass reserves, in report order.

Derived from the class so the reserved surface cannot drift from what it
protects. A field, attribute, or base-supplied member of the same name would
resolve ahead of the facade and leave the node's advertised surface unusable, so
each is rejected at class creation. Each name answers on the class as well as on
a value, which also exposes it to metaclass data-descriptor precedence, so a
metaclass binding of the same name is rejected alongside them.
"""


def _is_config_node(config_cls: Any) -> bool:
    """Report whether a class subclasses ``ConfigNode``.

    Args:
      config_cls (Any): The object to inspect.

    Returns:
      bool: True when ``config_cls`` is a class deriving from ``ConfigNode``.
    """
    return isinstance(config_cls, type) and issubclass(config_cls, ConfigNode)


def _missing_dataclass_message(config_cls: type[Any]) -> str:
    """Build the rejection message for a node class the ``@dataclass`` decorator skipped.

    Args:
      config_cls (type[Any]): The undecorated class.

    Returns:
      str: The rejection message naming the class and the required remedy.
    """
    name = _class_label(config_cls)
    if is_dataclass(config_cls):
        return (
            f"{name} declares annotations and subclasses ConfigNode, and its declaration did not apply "
            f"@dataclass, so those names stay outside the schema and only the inherited fields load. "
            f"Decorate {name} with @dataclass."
        )
    return (
        f"{name} subclasses ConfigNode without being a dataclass, so it carries no schema to build. "
        f"Decorate {name} with @dataclass."
    )
