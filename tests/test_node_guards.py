"""Guards a ``ConfigNode`` subclass carries beyond a plain dataclass.

Facade-name collisions and inherited equality ownership are rejected at class
creation; a skipped ``@dataclass`` decorator and a non-dataclass entry class are
reported as schema issues on the first engine operation.
"""

from __future__ import annotations

from dataclasses import (
    InitVar,
    dataclass,
    field,
    make_dataclass,
)
from typing import ClassVar

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    from_dict,
)
from confingo._node import _FACADE_NAMES


def _messages(error: ConfigError) -> str:
    return " | ".join(issue.message for issue in error.issues)


# ---------------------------------------------------------------------------
# G1: reserved facade names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_FACADE_NAMES))
def test_facade_name_as_annotated_field_is_rejected(name: str):
    """A field of a reserved name is rejected at class creation."""
    with pytest.raises(ConfigError) as error:
        make_dataclass("ShadowField", [(name, str, field(default="x"))], bases=(ConfigNode,))
    assert name in _messages(error.value)
    assert "shadows the ConfigNode method" in _messages(error.value)


@pytest.mark.parametrize("name", sorted(_FACADE_NAMES))
def test_facade_name_as_class_attribute_is_rejected(name: str):
    """A class-body binding of a reserved name is rejected at class creation."""
    with pytest.raises(ConfigError) as error:
        type("ShadowAttr", (ConfigNode,), {name: "x"})
    assert name in _messages(error.value)


@pytest.mark.parametrize("name", sorted(_FACADE_NAMES))
def test_facade_name_as_method_override_is_rejected(name: str):
    """A method override of a reserved name is rejected at class creation."""
    with pytest.raises(ConfigError) as error:
        type("ShadowMethod", (ConfigNode,), {name: lambda self: None})
    assert name in _messages(error.value)


def test_facade_name_inherited_as_a_field_is_rejected():
    """A reserved name inherited as a field from a plain base is rejected."""
    base = make_dataclass("PlainBaseWithToDict", [("cfg", str, field(default="x"))])
    with pytest.raises(ConfigError) as error:
        type("InheritsShadow", (base, ConfigNode), {})
    assert "is inherited as a field from PlainBaseWithToDict" in _messages(error.value)


def test_facade_name_inherited_as_a_field_is_rejected_with_the_node_base_first():
    """A field reaches the instance, so base order relative to ConfigNode is immaterial."""
    base = make_dataclass("PlainBaseFirstOrder", [("cfg", str, field(default="x"))])
    with pytest.raises(ConfigError) as error:
        make_dataclass("NodeBaseFirst", [("x", int, field(default=1))], bases=(ConfigNode, base))
    assert "is inherited as a field from PlainBaseFirstOrder" in _messages(error.value)


def test_facade_field_inherited_through_a_deeper_mro_is_rejected():
    """The field search runs base-ward, so a grandparent's field is still caught."""
    root = make_dataclass("GrandparentWithFacade", [("cfg", str, field(default="x"))])
    middle = make_dataclass("MiddleNode", [("m", int, field(default=1))], bases=(root,))
    with pytest.raises(ConfigError) as error:
        make_dataclass("DeepNode", [("d", int, field(default=2))], bases=(ConfigNode, middle))
    assert "is inherited as a field from GrandparentWithFacade" in _messages(error.value)


def test_facade_name_supplied_by_a_base_after_config_node_is_accepted():
    """A plain binding after ConfigNode in the MRO loses to the facade, so it is no collision."""
    mixin = type("LateLoaderMixin", (), {"cfg": lambda self: None})
    node = make_dataclass("NodeBeforeMixin", [("a", int, field(default=1))], bases=(ConfigNode, mixin))
    assert node.cfg._owner is node


def test_facade_name_supplied_by_an_earlier_base_is_rejected():
    """A base preceding ConfigNode that supplies a reserved name is rejected."""
    mixin = type("LoaderMixin", (), {"cfg": lambda self: None})
    with pytest.raises(ConfigError) as error:
        type("InheritsMixin", (mixin, ConfigNode), {})
    assert "is supplied by base LoaderMixin" in _messages(error.value)


def test_facade_name_in_own_slots_is_rejected():
    """A ``__slots__`` entry becomes a class-dict descriptor, which shadows the method."""
    with pytest.raises(ConfigError) as error:
        type("SlotShadow", (ConfigNode,), {"__slots__": ("cfg",)})
    assert "cfg" in _messages(error.value)


def test_facade_name_as_a_property_is_rejected():
    """A property of a reserved name is a class-body binding like any other."""
    with pytest.raises(ConfigError) as error:
        type("PropertyShadow", (ConfigNode,), {"cfg": property(lambda self: "x")})
    assert "cfg" in _messages(error.value)


def test_facade_name_in_a_later_base_slots_keeps_the_method():
    """A ``__slots__`` descriptor after ConfigNode loses to the facade in MRO order."""
    slot_base = type("LateSlotBase", (), {"__slots__": ("cfg",)})
    node = type("NodeBeforeSlots", (ConfigNode, slot_base), {})
    assert node().cfg is not None


def test_a_collision_names_the_accessor_and_the_way_it_was_taken():
    with pytest.raises(ConfigError) as error:
        make_dataclass(
            "ManyShadows",
            [("cfg", str, field(default="a"))],
            bases=(ConfigNode,),
        )
    assert len(error.value.issues) == 1
    assert "ManyShadows.cfg is declared as a field" in error.value.issues[0].message


def test_reserved_names_are_free_on_a_plain_dataclass():
    """A plain dataclass shadows no method, so the same names carry no restriction."""
    plain = make_dataclass("PlainWithFacadeNames", [("cfg", str, field(default="x"))])
    assert from_dict(plain, {"cfg": "y"}).cfg == "y"


@pytest.mark.parametrize("annotation", ["int", "ClassVar[int]", "InitVar[int]", "ClassVarLike[int]"])
def test_a_node_may_not_annotate_a_reserved_name_at_all(annotation: str):
    """Any annotation of a reserved name is rejected, whatever it would resolve to.

    Deciding whether a string annotation would become a stored field means
    reproducing the standard library's alias resolution, and a node has no
    reason to spell one of its own method names.
    """
    with pytest.raises(ConfigError) as error:
        type("OwnAnnotation", (ConfigNode,), {"__annotations__": {"cfg": annotation}})
    assert "is declared as a field" in _messages(error.value)


def test_inherited_class_var_of_a_reserved_name_is_still_accepted():
    """What a class inherits is judged by what lands on it, which fields() answers exactly."""
    node = make_dataclass("OverInheritedCV", [("y", int, field(default=2))], bases=(ConfigNode, ClassVarMetadataBase))
    assert node(y=2).cfg is not None


class RaisingDescriptorMeta(type):
    """Metaclass whose descriptor raises when resolution touches it."""

    @property
    def cfg(cls) -> str:
        raise RuntimeError("resolved during class creation")


def test_metaclass_descriptor_is_detected_without_being_run():
    """Descriptor kind is read from the class dict, so a getter with side effects never runs."""
    base = RaisingDescriptorMeta("RaisingBase", (), {})
    with pytest.raises(ConfigError) as error:
        type("RaisedOver", (ConfigNode, base), {})
    assert "metaclass RaisingDescriptorMeta" in _messages(error.value)


class SplitDescriptor:
    """Descriptor handing back the real method for class access and a string otherwise."""

    def __get__(self, instance: object, owner: type | None = None) -> object:
        return ConfigNode.__dict__["cfg"] if instance is None else "shadow"


def test_descriptor_matching_the_method_on_class_access_is_rejected():
    """Class access cannot establish instance behavior, so the binding alone rejects."""
    with pytest.raises(ConfigError) as error:
        type("SplitShadow", (ConfigNode,), {"cfg": SplitDescriptor()})
    assert "is bound in the class body" in _messages(error.value)


def test_staticmethod_wrapping_a_builder_function_is_rejected():
    """The underlying function matches, but a staticmethod drops the class binding."""
    builder = ConfigNode.__dict__["cfg"]
    mixin = type("StaticBuilderMixin", (), {"cfg": staticmethod(builder)})
    with pytest.raises(ConfigError) as error:
        make_dataclass("StaticShadow", [("x", int, field(default=1))], bases=(mixin, ConfigNode))
    assert "is supplied by base StaticBuilderMixin" in _messages(error.value)


def test_bound_builder_alias_from_config_node_is_rejected():
    """An alias stays bound to ConfigNode, so engine calls would target the wrong class."""
    mixin = type("AliasMixin", (), {"cfg": ConfigNode.__dict__["cfg"]})
    with pytest.raises(ConfigError) as error:
        make_dataclass("AliasShadow", [("x", int, field(default=1))], bases=(mixin, ConfigNode))
    assert "is supplied by base AliasMixin" in _messages(error.value)


def test_reserved_surface_is_one_accessor():
    assert set(_FACADE_NAMES) == {"cfg"}


# ---------------------------------------------------------------------------
# G2: inherited equality ownership
# ---------------------------------------------------------------------------


@dataclass
class HandWrittenEq:
    """Plain dataclass hand-writing equality, used as a base below."""

    a: int = 1

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = None  # type: ignore[assignment]


@dataclass
class HandWrittenHash:
    """Plain dataclass hand-writing hashing, used as a base below."""

    a: int = 1

    def __hash__(self) -> int:
        return 0


def test_undecorated_node_child_of_a_hand_written_eq_base_is_rejected():
    with pytest.raises(ConfigError) as error:
        type("UndecoratedKid", (HandWrittenEq, ConfigNode), {})
    assert "HandWrittenEq defines a custom __eq__" in _messages(error.value)


def test_decorated_node_child_of_a_hand_written_eq_base_is_rejected():
    """The guard cannot see whether ``@dataclass`` follows, so it covers both."""
    with pytest.raises(ConfigError) as error:
        make_dataclass("DecoratedKid", [("b", int, field(default=2))], bases=(HandWrittenEq, ConfigNode))
    assert "HandWrittenEq defines a custom __eq__" in _messages(error.value)


def test_node_child_of_a_hand_written_hash_base_is_rejected():
    with pytest.raises(ConfigError) as error:
        type("HashKid", (HandWrittenHash, ConfigNode), {})
    assert "HandWrittenHash defines a custom __hash__" in _messages(error.value)


def test_decorated_plain_child_of_a_hand_written_eq_base_is_accepted():
    """A plain dataclass generates its own ``__eq__``, which confingo then owns."""
    child = make_dataclass("PlainKid", [("b", int, field(default=2))], bases=(HandWrittenEq,))
    assert from_dict(child, {"b": 5}).b == 5


def test_node_subclassing_a_plain_dataclass_is_accepted():
    base = make_dataclass("OrdinaryBase", [("a", int, field(default=1))])
    node = make_dataclass("NodeOverOrdinary", [("b", int, field(default=2))], bases=(base, ConfigNode))
    assert node.cfg.from_dict({"a": 3}).a == 3


# ---------------------------------------------------------------------------
# G3: a subclass whose declaration skipped @dataclass
# ---------------------------------------------------------------------------


@dataclass
class DecoratedNode(ConfigNode):
    """Decorated node used as a base for the undecorated-subclass cases."""

    a: int = 1


class ForgotDecorator(DecoratedNode):
    """Subclass declaring a field without carrying ``@dataclass``."""

    b: int = 2


class ClassVarOnly(DecoratedNode):
    """Subclass declaring only a ``ClassVar``, which is never a field."""

    REGISTRY: ClassVar[dict[str, int]] = {}


class NoNewFields(DecoratedNode):
    """Subclass declaring no annotations of its own."""


def test_undecorated_subclass_declaring_a_field_is_rejected():
    with pytest.raises(ConfigError) as error:
        ForgotDecorator.cfg.from_dict({"a": 7})
    assert "did not apply @dataclass" in _messages(error.value)
    assert "ForgotDecorator" in _messages(error.value)


def test_undecorated_subclass_with_only_a_class_var_is_accepted():
    assert ClassVarOnly.cfg.from_dict({"a": 7}).a == 7


def test_undecorated_subclass_without_new_annotations_is_accepted():
    assert NoNewFields.cfg.from_dict({"a": 7}).a == 7


def test_decorated_subclass_is_accepted():
    decorated = make_dataclass("Decorated", [("b", int, field(default=2))], bases=(DecoratedNode,))
    assert decorated.cfg.from_dict({"a": 7, "b": 9}).b == 9


@dataclass
class HoldsForgotten(ConfigNode):
    """Enclosing node holding a section whose declaration skipped ``@dataclass``."""

    inner: ForgotDecorator


def test_undecorated_nested_node_is_reported_at_its_schema_path():
    with pytest.raises(ConfigError) as error:
        HoldsForgotten.cfg.from_dict({})
    assert [issue.path for issue in error.value.issues] == ["inner"]


# ---------------------------------------------------------------------------
# G4: entry class that is not a dataclass
# ---------------------------------------------------------------------------


class BareNode(ConfigNode):
    """ConfigNode subclass that never became a dataclass."""


class NotAConfig:
    """Plain class with no dataclass decoration."""


def test_non_dataclass_node_entry_is_rejected():
    with pytest.raises(ConfigError) as error:
        BareNode.cfg.from_dict({})
    assert "without being a dataclass" in _messages(error.value)


def test_non_dataclass_entry_is_rejected():
    with pytest.raises(ConfigError) as error:
        from_dict(NotAConfig, {})
    assert "is not a dataclass" in _messages(error.value)


def test_non_dataclass_entry_keeps_the_callers_context():
    with pytest.raises(ConfigError) as error:
        from_dict(NotAConfig, {}, context="config file train.json")
    assert error.value.context == "config file train.json"


# ---------------------------------------------------------------------------
# G1: shadowing routes that resolve outside the ordinary class MRO
# ---------------------------------------------------------------------------


class DataDescriptorMeta(type):
    """Metaclass supplying a builder name as a data descriptor."""

    @property
    def cfg(cls) -> str:
        return "shadow"


class NonDataDescriptorMeta(type):
    """Metaclass supplying a builder name as an ordinary method."""

    def cfg(cls) -> str:
        return "shadow"


def test_metaclass_data_descriptor_over_a_classmethod_is_rejected():
    """A metaclass data descriptor outranks the class MRO, so it replaces the builder."""
    base = DataDescriptorMeta("DescriptorBase", (), {})
    with pytest.raises(ConfigError) as error:
        make_dataclass("MetaShadowed", [("x", int, field(default=1))], bases=(ConfigNode, base))
    assert "data descriptor by metaclass DataDescriptorMeta" in _messages(error.value)


def test_metaclass_non_data_binding_keeps_the_classmethod():
    """A non-data metaclass binding loses to the class MRO, so it is no collision."""
    base = NonDataDescriptorMeta("PlainMetaBase", (), {})
    node = make_dataclass("MetaIntact", [("x", int, field(default=1))], bases=(ConfigNode, base))
    assert node.cfg._owner is node


@dataclass
class ClassVarMetadataBase:
    """Plain dataclass declaring a reserved name as a ``ClassVar`` rather than a field."""

    cfg: ClassVar[str] = "metadata"
    x: int = 1


@dataclass
class FieldThenClassVarBase:
    """Plain dataclass declaring a reserved name as a stored field."""

    cfg: str = "old field"
    x: int = 1


@dataclass
class ClassVarOverrideBase(FieldThenClassVarBase):
    """Subclass that redeclares the inherited field as a ``ClassVar``, removing it."""

    # Narrowing an inherited field to a ClassVar is exactly the removal this
    # fixture exercises; dataclasses honor it, so the schema no longer stores it.
    cfg: ClassVar[str] = "metadata"  # pyrefly: ignore[bad-override]


def test_inherited_class_var_pseudo_field_is_accepted():
    """``ClassVar`` entries are not stored on the instance, so they shadow nothing."""
    node = make_dataclass("OverClassVar", [("y", int, field(default=2))], bases=(ConfigNode, ClassVarMetadataBase))
    assert node(y=2).cfg is not None


def test_field_removed_by_a_class_var_override_is_accepted():
    """A base that redeclares an inherited field as a ``ClassVar`` removes it from the schema."""
    node = make_dataclass("OverRemoved", [("y", int, field(default=2))], bases=(ConfigNode, ClassVarOverrideBase))
    assert node(y=2).cfg is not None


def test_stored_field_reintroduced_after_a_class_var_override_is_rejected():
    """The most-derived declaration decides, so a restored stored field still collides."""
    restored = make_dataclass("RestoredField", [("cfg", str, field(default="back"))], bases=(ClassVarOverrideBase,))
    with pytest.raises(ConfigError) as error:
        make_dataclass("OverRestored", [("y", int, field(default=2))], bases=(ConfigNode, restored))
    assert "is inherited as a field from RestoredField" in _messages(error.value)


@dataclass
class InitVarNode(ConfigNode):
    """Decorated node carrying an ``InitVar``, which is never a schema field."""

    seed: InitVar[int] = 0
    a: int = 1

    def __post_init__(self, seed: int) -> None:
        self.a = self.a + seed


class UndecoratedInitVar(InitVarNode):
    """Undecorated subclass declaring an ``InitVar`` of its own."""

    scale: InitVar[int] = 2


def test_decorated_node_with_an_init_var_is_accepted():
    assert InitVarNode.cfg.from_dict({"a": 5}).a == 5


def test_undecorated_subclass_declaring_an_init_var_is_rejected():
    """An ``InitVar`` needs the decorator to mean anything, so the class is reported."""
    with pytest.raises(ConfigError) as error:
        UndecoratedInitVar.cfg.from_dict({"a": 5})
    assert "did not apply @dataclass" in _messages(error.value)


class HidingType(type):
    """Metaclass denying the descriptor slots, recording each probe it sees."""

    probes: ClassVar[list[str]] = []

    def __getattribute__(cls, name: str) -> object:
        if name in {"__set__", "__delete__"}:
            HidingType.probes.append(name)
            raise AttributeError(name)
        return super().__getattribute__(name)


class HiddenDataDescriptor(metaclass=HidingType):
    """Data descriptor whose type denies ``__set__`` to ordinary attribute lookup."""

    def __get__(self, instance: object, owner: type | None = None) -> str:
        return "shadow"

    def __set__(self, instance: object, value: object) -> None:
        raise RuntimeError("not used")


def test_data_descriptor_hidden_from_attribute_lookup_is_still_detected():
    """Descriptor kind is read from raw namespaces, so a denying metaclass cannot hide it."""
    HidingType.probes.clear()
    descriptor_meta = type("HidingDescriptorMeta", (type,), {"cfg": HiddenDataDescriptor()})
    base = descriptor_meta("HidingDescriptorBase", (), {})
    with pytest.raises(ConfigError) as error:
        type("HiddenShadow", (ConfigNode, base), {})
    assert "metaclass HidingDescriptorMeta" in _messages(error.value)
    assert HidingType.probes == []
