"""Tests for the supported-type boundary and the messages that name it.

One builder produces the type-boundary message, so every route to an unsupported
annotation names the same supported categories and the same runtime-state remedy.
A class that reads like a section whose declaration skipped ``@dataclass`` routes
to the decorator remedy instead.
"""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003  (needed at runtime by get_type_hints)
from dataclasses import (
    dataclass,
    field,
)
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Generic,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
    override,
)

import pytest

from confingo import (
    ConfigError,
    ConfigValue,
)
from confingo.functional import (
    from_dict,
    to_dict,
)


BOUNDARY_REMEDY = (
    "choose a supported annotation (bool, int, float, str, Path, date/time, Enum/Literal, dataclass, "
    "container/union, array/tensor, or ConfigValue/ConfigScalar for plain data) and derive other runtime "
    "values in an init=False field"
)

DECORATOR_REMEDY = "is not a dataclass, so it carries no config schema. Declare it with @dataclass."


def _messages(config_cls: type[Any], data: dict[str, Any] | None = None) -> dict[str, str]:
    """Build ``config_cls`` and return its reported issues keyed by path.

    Args:
      config_cls (type[Any]): The schema class expected to fail.
      data (dict[str, Any] | None = None): The mapping to build from, empty by default.

    Returns:
      dict[str, str]: One message per reported path.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {} if data is None else data)
    return {issue.path: issue.message for issue in info.value.issues}


@dataclass
class HasDecimal:
    amount: Decimal = Decimal(0)


def test_unsupported_leaf_type_names_the_boundary_and_the_remedy():
    assert _messages(HasDecimal, {"amount": "1.5"}) == {"amount": f"unsupported field type Decimal; {BOUNDARY_REMEDY}"}


class Section(TypedDict):
    a: int


@dataclass
class HasTypedDict:
    section: Section = field(default_factory=lambda: {"a": 1})


class Point(NamedTuple):
    x: int


@dataclass
class HasNamedTuple:
    point: Point = field(default_factory=lambda: Point(1))


@dataclass
class HasUnsupportedGeneric:
    items: Iterable[int] = ()


@dataclass
class Nested:
    inner: HasDecimal = field(default_factory=HasDecimal)


@pytest.mark.parametrize(
    ("config_cls", "path", "rendered"),
    [
        (HasTypedDict, "section", "Section"),
        (HasNamedTuple, "point", "Point"),
        (HasUnsupportedGeneric, "items", "Iterable[int]"),
        (Nested, "inner.amount", "Decimal"),
    ],
)
def test_every_route_to_the_boundary_shares_one_message(config_cls: type[Any], path: str, rendered: str):
    # A TypedDict and a NamedTuple carry their own annotations without being
    # dataclasses, and each is a deliberate choice rather than a forgotten
    # decorator, so both stay on the type-boundary message.
    assert _messages(config_cls)[path] == f"unsupported field type {rendered}; {BOUNDARY_REMEDY}"


@dataclass
class AnyField:
    x: ConfigValue = None


def test_to_dict_rejects_unserializable_value():
    # Decimal sits outside ConfigValue; the marshal gate is what this exercises.
    cfg = AnyField(x=Decimal("1.5"))  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(ConfigError) as info:
        to_dict(cfg)
    assert "Decimal" in str(info.value)


def test_any_field_still_passes_plain_data():
    cfg = AnyField(x=[1, "two", {"three": 3}])
    assert to_dict(cfg) == {"x": [1, "two", {"three": 3}]}


# --- a class that forgot @dataclass ------------------------------------------


class Forgot:
    host: str = "localhost"
    port: int = 8080


class NoAnnotations:
    pass


@dataclass
class DirectlyNested:
    server: Forgot = None  # pyrefly: ignore[bad-assignment]


@dataclass
class DepthTwo:
    inner: DirectlyNested = field(default_factory=DirectlyNested)


@dataclass
class InContainer:
    servers: list[Forgot] = field(default_factory=list)


@dataclass
class InUnion:
    server: Forgot | None = None


@dataclass
class InMappingValue:
    servers: dict[str, Forgot] = field(default_factory=dict)


@dataclass
class WithoutAnnotations:
    thing: NoAnnotations = None  # pyrefly: ignore[bad-assignment]


@pytest.mark.parametrize(
    ("config_cls", "path"),
    [
        (DirectlyNested, "server"),
        (DepthTwo, "inner.server"),
        (InContainer, "servers"),
        (InUnion, "server"),
        (InMappingValue, "servers"),
    ],
)
def test_an_undecorated_schema_class_gets_the_decorator_remedy(config_cls: type[Any], path: str):
    assert _messages(config_cls)[path] == f"Forgot {DECORATOR_REMEDY}"


def test_the_nested_message_matches_the_entry_message():
    nested = _messages(DirectlyNested)["server"]
    entry = _messages(Forgot)[""]
    assert nested == entry


def test_a_class_without_annotations_stays_on_the_type_boundary():
    assert _messages(WithoutAnnotations)["thing"] == f"unsupported field type NoAnnotations; {BOUNDARY_REMEDY}"


@dataclass
class TwoProblems:
    server: Forgot = None  # pyrefly: ignore[bad-assignment]
    amount: Decimal = Decimal(0)


def test_boundary_and_decorator_issues_aggregate():
    reported = _messages(TwoProblems)
    assert reported["server"] == f"Forgot {DECORATOR_REMEDY}"
    assert reported["amount"] == f"unsupported field type Decimal; {BOUNDARY_REMEDY}"


# --- classes whose annotations belong to something else -----------------------


EVALUATED: list[str] = []


def _mark() -> type:
    """Record that an annotation expression was evaluated.

    Returns:
      type: A throwaway annotation type.
    """
    EVALUATED.append("mark")
    return int


class DataProtocol(Protocol):
    host: str


class Box[T]:
    value: T


class ForeignModel:
    __attrs_attrs__: ClassVar[tuple[Any, ...]] = ()
    host: str


class Constants:
    MAX: ClassVar[int] = 10


class UnresolvableConstant:
    MAX: ClassVar[Missing]  # noqa: F821  # pyrefly: ignore[unknown-name]  (deliberately unresolvable)


class SideEffectAnnotation:
    # Stored as the source text "_mark()"; evaluating it would append to EVALUATED.
    # pyrefly: ignore[invalid-annotation]
    value: _mark()


@dataclass
class HasProtocol:
    server: DataProtocol = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasGeneric:
    box: Box = None  # pyrefly: ignore[bad-assignment, implicit-any-type-argument]


@dataclass
class HasForeignModel:
    model: ForeignModel = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasConstants:
    limits: Constants = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasUnresolvableConstant:
    limits: UnresolvableConstant = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasSideEffectAnnotation:
    thing: SideEffectAnnotation = None  # pyrefly: ignore[bad-assignment]


@pytest.mark.parametrize(
    ("config_cls", "path", "rendered"),
    [
        (HasProtocol, "server", "DataProtocol"),
        (HasGeneric, "box", "Box"),
        (HasForeignModel, "model", "ForeignModel"),
        (HasConstants, "limits", "Constants"),
        (HasUnresolvableConstant, "limits", "UnresolvableConstant"),
    ],
)
def test_deliberate_declarations_stay_on_the_type_boundary(config_cls: type[Any], path: str, rendered: str):
    # A protocol, a generic class carrying type parameters, a model belonging to
    # another schema system, and class constants each declare annotations for a
    # reason of their own, so the decorator remedy would be wrong advice.
    assert _messages(config_cls)[path] == f"unsupported field type {rendered}; {BOUNDARY_REMEDY}"


def test_preflight_does_not_evaluate_an_unsupported_class_annotation():
    EVALUATED.clear()
    reported = _messages(HasSideEffectAnnotation)
    assert reported["thing"] == f"SideEffectAnnotation {DECORATOR_REMEDY}"
    assert EVALUATED == []


class RaisingMeta(type):
    """A metaclass whose attribute hook fails, to prove classification stays clear of it."""

    def __getattr__(cls, name: str) -> Any:
        """Fail on every dynamic attribute lookup.

        Args:
          name (str): The attribute being looked up.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError(f"metaclass hook invoked for {name}")


class HostileClass(metaclass=RaisingMeta):
    host: str = "localhost"


class HostileNoAnnotations(metaclass=RaisingMeta):
    pass


@dataclass
class HasHostile:
    server: HostileClass = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasHostileWithoutAnnotations:
    thing: HostileNoAnnotations = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HostileClassUnderAny:
    # A class object sits outside ConfigValue; the schema is ill-typed on purpose
    # so the authored-default gate is the thing that judges it.
    thing: ConfigValue = HostileClass  # pyrefly: ignore[bad-assignment]


def test_judging_a_class_object_reports_it_instead_of_waking_its_metaclass():
    # The plain-domain gate judges the class object by its type alone, so the
    # metaclass hook stays dormant and the author gets a ConfigError naming the
    # metaclass that produced the value.
    assert _messages(HostileClassUnderAny)["thing"] == (
        "invalid authored default: expected plain data for ConfigValue, got RaisingMeta; "
        "use a scalar, a list, or a str-keyed mapping, or name the type with a dataclass section"
    )


def test_classifying_a_field_type_never_invokes_its_metaclass_hook():
    # Naming a class as an annotation is the only thing the author did, so
    # classifying it reads raw namespaces; an attribute lookup here would raise
    # RuntimeError straight past the collector.
    assert _messages(HasHostile)["server"] == f"HostileClass {DECORATOR_REMEDY}"
    assert (
        _messages(HasHostileWithoutAnnotations)["thing"]
        == f"unsupported field type HostileNoAnnotations; {BOUNDARY_REMEDY}"
    )


class FieldsMarker:
    _fields: ClassVar[tuple[Any, ...]] = ()


class ForgotWithFieldsMarker(FieldsMarker):
    host: str = "localhost"


@dataclass
class HasFieldsMarkerClass:
    server: ForgotWithFieldsMarker = None  # pyrefly: ignore[bad-assignment]


def test_a_plain_class_binding_fields_still_gets_the_decorator_remedy():
    # _fields identifies a NamedTuple only alongside the tuple base; an ordinary
    # class that happens to bind the name is still a forgotten schema.
    assert _messages(HasFieldsMarkerClass)["server"] == f"ForgotWithFieldsMarker {DECORATOR_REMEDY}"


class SeparatedInt(int):
    """An int subclass whose instances stay distinct from one another.

    Comparing against another instance answers by identity, so two instances
    built from the same number are distinct enum member values rather than
    aliases; comparing against an ordinary int answers as int does, so the
    member lookup a load performs still reaches one of them.
    """

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeparatedInt):
            return self is other
        return int.__eq__(self, other)

    __hash__ = int.__hash__


class SplitValued(Enum):
    # Two members carrying the same number is what this fixture exists to build:
    # SeparatedInt keeps them distinct members rather than collapsing B into an
    # alias of A, which is the shape preflight is asked to report.
    A = SeparatedInt(1)
    B = SeparatedInt(1)  # noqa: PIE796


class TextSubclass(str):
    pass


class TextValued(Enum):
    A = TextSubclass("a")


class TupleValued(Enum):
    A = (1, 2)


@dataclass
class HasSplitValued:
    choice: SplitValued = SplitValued.A


@dataclass
class HasSplitValuedSet:
    choices: set[SplitValued] = field(default_factory=set)


@dataclass
class HasTextValued:
    choice: TextValued = TextValued.A


@dataclass
class HasTupleValued:
    choice: TupleValued = TupleValued.A


ENUM_VALUE_REMEDY = "give each member an exact bool, int, or str value"


def test_two_members_separated_only_by_a_subclass_are_reported_at_preflight():
    # Both members write 1, and the value a file carries rebuilds whichever the
    # lookup reaches first, so a set of the two would rebuild as one member.
    # Preflight names the subclass and asks for the exact int instead.
    message = _messages(HasSplitValued)["choice"]
    assert "enum SplitValued must carry primitive values" in message
    assert "int subclass SeparatedInt" in message
    assert message.endswith(ENUM_VALUE_REMEDY)


def test_the_subclass_rule_reaches_an_enum_named_as_a_set_element():
    # The set element route reads the same member values, so the collapse is
    # reported where it would happen rather than only on a scalar field.
    message = _messages(HasSplitValuedSet)["choices"]
    assert "enum SplitValued must carry primitive values" in message
    assert message.endswith(ENUM_VALUE_REMEDY)


def test_a_str_subclass_member_value_is_reported_even_when_it_stands_alone():
    # One member cannot collide with a sibling, and the rule still holds: a load
    # builds the exact str a file carries, which is a different value than the
    # subclass the member declares.
    message = _messages(HasTextValued)["choice"]
    assert "str subclass TextSubclass" in message
    assert message.endswith(ENUM_VALUE_REMEDY)


def test_a_member_value_outside_the_primitives_names_what_a_load_reads():
    # A tuple is a form the member lookup is never handed, so the message names
    # the three classes a file carries rather than a subclass relationship.
    message = _messages(HasTupleValued)["choice"]
    assert "TupleValued must carry primitive values" in message
    assert "a load reads bool, int, or str from a file" in message
    assert message.endswith(ENUM_VALUE_REMEDY)


class Plain(Enum):
    NAME = "plain"
    COUNT = 3
    FLAG = True


@dataclass
class HasPlain:
    by_value: Plain
    by_name: Plain
    by_number: Plain


def test_exact_primitive_member_values_stay_admitted():
    # The rule reaches only the subclass, so the ordinary enum a schema declares
    # keeps building and rendering.
    built = from_dict(HasPlain, {"by_value": "plain", "by_name": "COUNT", "by_number": True})
    assert (built.by_value, built.by_name, built.by_number) == (Plain.NAME, Plain.COUNT, Plain.FLAG)
    assert to_dict(built) == {"by_value": "plain", "by_name": 3, "by_number": True}


class UnnameableMeta(type):
    """A metaclass that raises when its class is named.

    Intercepting ``__getattribute__`` is what makes the read fail while the
    declaration itself stays a valid one a type checker accepts, which is the
    boundary these tests exist to show: the failure is confingo's to contain
    because no checker can rule it out.
    """

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("name boom")
        return super().__getattribute__(name)


class UnnameableLeaf(metaclass=UnnameableMeta):
    pass


@dataclass
class HasUnnameableLeafValue:
    x: int = 0


def test_a_supplied_value_whose_class_declines_to_be_named_is_reported_by_its_path():
    # Naming the value's class is what the mismatch message does, so the class
    # answering with a failure costs the name rather than the report.
    assert _messages(HasUnnameableLeafValue, {"x": UnnameableLeaf()})["x"] == (
        "expected int, got a class that could not be named"
    )


NAME_ARMED = [False]
"""Whether the toggled metaclass below refuses to answer for its class name.

A one-element list rather than a bare name, so arming it is a mutation of the
container rather than a rebinding the test would have to declare global.
"""


class ToggledNameMeta(type):
    """A metaclass whose class name becomes unreadable once the flag is set.

    ``@dataclass`` reads the name it decorates, so the failure is armed after
    decoration; from then on the class is an ordinary schema whose name no
    message can read.
    """

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__" and NAME_ARMED[0]:
            raise RuntimeError("flag check name boom")
        return super().__getattribute__(name)


@dataclass(frozen=True)
class TogglesItsName(metaclass=ToggledNameMeta):
    x: int = 1


def test_the_flag_check_names_its_class_through_the_guarded_read():
    # _flag_conflicts names the class for every message it may raise, and it runs
    # for every schema, so an unreadable name there would fail every load rather
    # than one message. The build succeeds and reads the value.
    NAME_ARMED[0] = True
    try:
        assert from_dict(TogglesItsName, {"x": 2}).x == 2
    finally:
        NAME_ARMED[0] = False


ElementT = TypeVar("ElementT")


@dataclass
class LegacyGenericSchema(Generic[ElementT]):  # noqa: UP046  (the legacy spelling is the fixture)
    count: int = 0


@dataclass
class LegacyGenericChild(LegacyGenericSchema[int]):
    extra: int = 1


class LegacyProtocolBase(Protocol[ElementT]):
    pass


@dataclass
class LegacyProtocolChild(LegacyProtocolBase[int]):
    count: int = 0


class ParameterizedMeta[ElementP](type):
    pass


@dataclass
class HasGenericMetaclass(metaclass=ParameterizedMeta[int]):
    count: int = 0


@dataclass
class Pep695Schema[ElementP]:
    count: int = 0


@dataclass
class AliasGenericSchema(list[ElementT]):
    total: int = 0


@dataclass
class AliasGenericChild(AliasGenericSchema[int]):
    extra: int = 1


@pytest.mark.parametrize(
    ("config_cls", "owner", "parameter"),
    [
        (Pep695Schema, "Pep695Schema", "ElementP"),
        (LegacyGenericSchema, "LegacyGenericSchema", "ElementT"),
        (LegacyGenericChild, "LegacyGenericSchema", "ElementT"),
        (LegacyProtocolChild, "LegacyProtocolBase", "ElementT"),
        (AliasGenericChild, "AliasGenericSchema", "ElementT"),
    ],
    ids=["pep-695", "legacy-generic", "legacy-generic-base", "legacy-protocol-base", "parameterized-alias-base"],
)
def test_a_schema_owning_type_parameters_is_reported_whichever_way_it_declares_them(
    config_cls: type[Any], owner: str, parameter: str
):
    # A config file carries concrete values, so a load builds the type an
    # annotation names and a parameter names none. PEP 695 records parameters
    # under __type_params__ and the legacy spellings under __parameters__, and a
    # generic metaclass owns them from outside the class's own MRO; the class
    # that declares them is the one reported in every case.
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {})
    (issue,) = info.value.issues
    assert issue.message.startswith(f"{owner} takes the type parameter {parameter},")
    assert issue.message.endswith("derive anything that varies in an init=False field")


def test_a_generic_metaclass_owner_gets_a_metaclass_remedy():
    # The section already writes concrete field types, and the parameter may sit
    # behind a library's exported metaclass alias, so the remedy names the
    # metaclass rather than telling the author to write out types they wrote.
    with pytest.raises(ConfigError) as info:
        from_dict(HasGenericMetaclass, {})
    (issue,) = info.value.issues
    assert issue.message.startswith(
        "ParameterizedMeta, the metaclass of HasGenericMetaclass, takes the type parameter ElementP,"
    )
    assert issue.message.endswith("build HasGenericMetaclass with a metaclass that takes none")


@dataclass
class ConcreteGenericBase(dict[str, int]):
    count: int = 0


class OrdinaryMeta(type):
    """A metaclass declaring no type parameters of its own."""


@dataclass
class OrdinaryMetaclassSchema(metaclass=OrdinaryMeta):
    count: int = 0


def test_a_schema_under_an_ordinary_metaclass_keeps_building():
    # An ordinary metaclass declares no parameters, so the rule leaves it alone.
    assert from_dict(OrdinaryMetaclassSchema, {"count": 3}).count == 3


def test_a_base_specialized_to_concrete_types_reports_the_shape_rather_than_a_parameter():
    # dict[str, int] names concrete types, so the type-parameter rule stays quiet.
    # What answers instead is the rule against a section that is also one of the
    # kinds every walk dispatches on, which a dict base makes this class.
    with pytest.raises(ConfigError) as info:
        from_dict(ConcreteGenericBase, {"count": 3})
    (issue,) = info.value.issues
    assert issue.message.startswith("ConcreteGenericBase is a config section and also a mapping")
    assert "takes the type parameter" not in issue.message
