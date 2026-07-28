"""Tests for the preflight rule on what a ``set`` or ``frozenset`` element may name.

A set holds its elements by hash and a load rebuilds each one from plain data, so
an element annotation is admitted when every value it accepts rebuilds hashable:
scalars, and ``tuple`` / ``frozenset`` shapes over scalars. Config dataclasses are
unhashable and keep a remedy of their own naming ``config_hash``. Sets of hashable
values keep their deduplication, deterministic serialization, and round trips.
"""

from __future__ import annotations

import datetime as dt
from collections import deque  # noqa: TC003  (needed at runtime by get_type_hints)
from dataclasses import (
    dataclass,
    field,
    make_dataclass,
)
from enum import (
    Enum,
    EnumType,
    IntEnum,
    StrEnum,
)
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
    override,
)

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    ConfigValue,
)
from confingo.functional import (
    from_dict,
    to_dict,
)


@dataclass
class Section:
    lr: float = 1e-3


@dataclass
class NodeSection(ConfigNode):
    lr: float = 1e-3


@dataclass
class BadSection:
    lr: complex = 0j


# --- rejected shapes ----------------------------------------------------------


@dataclass
class DirectFrozenset:
    sections: frozenset[Section] = field(default_factory=frozenset)


@dataclass
class DirectSet:
    sections: set[Section] = field(default_factory=set)


@dataclass
class NodeElementSet:
    sections: set[NodeSection] = field(default_factory=set)


@dataclass
class UnionElementSet:
    sections: set[Section | int] = field(default_factory=set)


@dataclass
class TupleShapeSet:
    sections: set[tuple[str, Section]] = field(default_factory=set)


@dataclass
class NestedFrozensetSet:
    sections: frozenset[frozenset[Section]] = field(default_factory=frozenset)


@dataclass
class MalformedSectionSet:
    sections: frozenset[BadSection] = field(default_factory=frozenset)


def _issues(config_cls: type[Any]) -> list[tuple[str, str]]:
    """Build ``config_cls`` from an empty mapping and return its reported issues.

    Args:
      config_cls (type[Any]): The schema class expected to fail preflight.

    Returns:
      list[tuple[str, str]]: One ``(path, message)`` pair per reported issue.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {})
    return [(issue.path, issue.message) for issue in info.value.issues]


def test_frozenset_of_sections_is_rejected_with_both_remedies():
    with pytest.raises(ConfigError) as info:
        from_dict(DirectFrozenset, {}, context="training.yaml")
    assert info.value.context == "training.yaml"
    assert [(issue.path, issue.message) for issue in info.value.issues] == [
        (
            "sections",
            "config sections are unhashable, so frozenset[Section] cannot be built; use a list or tuple "
            "for the collection, and use confingo.functional.config_hash(section) as the value-identity "
            "key when uniqueness matters",
        )
    ]


@pytest.mark.parametrize(
    ("config_cls", "rendered"),
    [
        (DirectSet, "set[Section]"),
        (NodeElementSet, "set[NodeSection]"),
        (UnionElementSet, "set[Section | int]"),
        (TupleShapeSet, "set[tuple[str, Section]]"),
        (NestedFrozensetSet, "frozenset[Section]"),
    ],
)
def test_section_bearing_sets_are_rejected_naming_the_written_annotation(config_cls: type[Any], rendered: str):
    reported = _issues(config_cls)
    assert all(path == "sections" for path, _message in reported), reported
    assert any(f"so {rendered} cannot be built" in message for _path, message in reported), reported


def test_a_nested_frozenset_names_the_annotation_that_cannot_be_built():
    """A frozenset hashes its members as it is built, so a bad member stops it there.

    The enclosing frozenset would hold frozensets, which are hashable, so the
    inner annotation is the one to change and the only one reported.
    """
    messages = [message for _path, message in _issues(NestedFrozensetSet)]
    assert len(messages) == 1, messages
    assert "so frozenset[Section] cannot be built" in messages[0], messages


def test_a_defect_inside_the_section_still_aggregates():
    messages = [message for _path, message in _issues(MalformedSectionSet)]
    assert any("so frozenset[BadSection] cannot be built" in message for message in messages)
    assert any("unsupported field type complex" in message for message in messages)


# --- elements whose rebuild is not hashable -----------------------------------


@dataclass
class AnySet:
    values: set[ConfigValue] = field(default_factory=set)


@dataclass
class BareSet:
    values: set = field(default_factory=set)  # pyrefly: ignore[implicit-any-type-argument]


@dataclass
class ListElementSet:
    values: set[list[int]] = field(default_factory=set)


@dataclass
class ScalarListUnionSet:
    values: set[int | list[int]] = field(default_factory=set)


@dataclass
class NestedAnyTupleSet:
    values: set[tuple[int, ConfigValue]] = field(default_factory=set)


@dataclass
class MappingElementSet:
    values: frozenset[dict[str, int]] = field(default_factory=frozenset)


@dataclass
class LeadingAnyTupleSet:
    values: set[tuple[Any, int]] = field(default_factory=set)


@dataclass
class EmptyArgsSet:
    values: set[()] = field(default_factory=set)  # pyrefly: ignore[bad-specialization]


@dataclass
class EmptyArgsFrozenset:
    values: frozenset[()] = field(default_factory=frozenset)  # pyrefly: ignore[bad-specialization]


@dataclass
class NestedEmptyArgsSet:
    values: set[frozenset[()]] = field(default_factory=set)  # pyrefly: ignore[bad-specialization]


class SuppressedHash(Enum):
    A = "a"
    __hash__: ClassVar[None] = None


class RaisingHash(Enum):
    A = "a"

    def __hash__(self) -> int:
        """Fail, so a preflight that relies on calling it is visible.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("hash hook ran")


@dataclass
class SuppressedHashSet:
    values: set[SuppressedHash] = field(default_factory=set)


@dataclass
class RaisingHashSet:
    values: set[RaisingHash] = field(default_factory=set)


@pytest.mark.parametrize(
    ("config_cls", "rendered", "element"),
    [
        (AnySet, "set[ConfigValue]", "ConfigValue"),
        (ListElementSet, "set[list[int]]", "list[int]"),
        (ScalarListUnionSet, "set[int | list[int]]", "int | list[int]"),
        (NestedAnyTupleSet, "set[tuple[int, ConfigValue]]", "tuple[int, ConfigValue]"),
        (LeadingAnyTupleSet, "set[tuple[Any, int]]", "tuple[Any, int]"),
        (MappingElementSet, "frozenset[dict[str, int]]", "dict[str, int]"),
        (SuppressedHashSet, "set[SuppressedHash]", "SuppressedHash"),
        (RaisingHashSet, "set[RaisingHash]", "RaisingHash"),
    ],
    ids=[
        "any",
        "list",
        "scalar-list-union",
        "nested-any-tuple",
        "leading-any-tuple",
        "mapping",
        "suppressed-enum-hash",
        "raising-enum-hash",
    ],
)
def test_elements_that_rebuild_unhashable_are_rejected_at_preflight(config_cls: type[Any], rendered: str, element: str):
    messages = [message for _path, message in _issues(config_cls)]
    assert any(f"{rendered} cannot be built" in message for message in messages), messages
    assert any(element in message for message in messages), messages
    assert any("hold the values in a list" in message for message in messages), messages


@dataclass
class ScalarUnionSet:
    values: set[int] = field(default_factory=set)


@dataclass
class ScalarTupleSet:
    values: set[tuple[str, int]] = field(default_factory=set)


@dataclass
class NestedScalarFrozensetSet:
    values: set[frozenset[str]] = field(default_factory=set)


@dataclass
class OptionalScalarSet:
    values: set[int | None] = field(default_factory=set)


@pytest.mark.parametrize(
    "config_cls",
    [ScalarUnionSet, ScalarTupleSet, NestedScalarFrozensetSet, OptionalScalarSet],
    ids=["scalar", "scalar-tuple", "nested-frozenset", "optional-scalar"],
)
def test_elements_that_rebuild_hashable_are_admitted(config_cls: type[Any]):
    assert from_dict(config_cls, {}).values == set()


# --- shapes that keep their current behavior ----------------------------------


@dataclass
class ScalarSets:
    tags: set[str] = field(default_factory=set)
    seeds: frozenset[int] = field(default_factory=frozenset)
    mixed: set[int] = field(default_factory=set)
    literals: set[Literal["on", "off"]] = field(default_factory=set)
    paired: set[tuple[str, int]] = field(default_factory=set)


def test_scalar_sets_build_and_deduplicate():
    built = from_dict(
        ScalarSets,
        {
            "tags": ["b", "a", "b"],
            "seeds": [2, 1, 2],
            "mixed": [1, 2, 1],
            "literals": ["on", "on", "off"],
            "paired": [["a", 1], ["a", 1], ["b", 2]],
        },
    )
    assert built.tags == {"a", "b"}
    assert built.seeds == frozenset({1, 2})
    assert built.mixed == {1, 2}
    assert built.literals == {"on", "off"}
    assert built.paired == {("a", 1), ("b", 2)}


def test_scalar_sets_serialize_deterministically_and_round_trip():
    built = from_dict(ScalarSets, {"tags": ["b", "a"], "seeds": [2, 1]})
    exported = to_dict(built)
    assert exported["tags"] == ["a", "b"]
    assert exported["seeds"] == [1, 2]
    assert from_dict(ScalarSets, exported) == built


# --- supported section collections --------------------------------------------


@dataclass
class SectionCollections:
    listed: list[Section] = field(default_factory=list)
    tupled: tuple[Section, ...] = ()


def test_lists_and_tuples_of_sections_remain_supported():
    built = from_dict(SectionCollections, {"listed": [{"lr": 0.1}], "tupled": [{"lr": 0.2}]})
    assert built.listed == [Section(lr=0.1)]
    assert built.tupled == (Section(lr=0.2),)
    assert from_dict(SectionCollections, to_dict(built)) == built


# --- exactly one message per rejected element --------------------------------


@dataclass
class UnsupportedGenericElementSet:
    values: set[deque[int]] = field(default_factory=set)


@dataclass
class BareElementSet:
    values: set[set] = field(default_factory=set)  # pyrefly: ignore[implicit-any-type-argument]


def test_an_unsupported_element_reports_only_the_unsupported_type():
    # confingo has no construction semantics for the annotation, so what it would
    # rebuild is undefined and the boundary message is the whole finding.
    messages = [message for _path, message in _issues(UnsupportedGenericElementSet)]
    assert messages == [
        "unsupported field type deque[int]; choose a supported annotation (bool, int, float, str, Path, "
        "date/time, Enum/Literal, dataclass, container/union, array/tensor, or ConfigValue/ConfigScalar for "
        "plain data) and derive other runtime values in an init=False field"
    ]


def test_a_bare_container_element_reports_its_own_and_the_container_instability():
    # An argument-free element names no contents, and it still rebuilds a mutable
    # set whatever those contents turn out to be, so the container's own
    # instability is established on its own and stands beside the element's remedy.
    messages = [message for _path, message in _issues(BareElementSet)]
    assert messages == [
        "set carries no element type; write set[ConfigScalar] for plain data of any shape, or name the element type",
        "set[set] cannot be built: a set element must rebuild hashable when a file is loaded, and set rebuilds "
        "a value a set cannot hold; use a scalar element, a tuple of scalars, or hold the values in a list",
    ], messages


def test_a_section_element_still_aggregates_defects_inside_the_section():
    messages = [message for _path, message in _issues(MalformedSectionSet)]
    assert any("so frozenset[BadSection] cannot be built" in message for message in messages), messages
    assert any("unsupported field type complex" in message for message in messages), messages


class Colour(Enum):
    RED = "red"


class Level(IntEnum):
    LOW = 1


class Mode(StrEnum):
    FAST = "fast"


# --- every supported scalar hashes by an implementation confingo relies on ----


@dataclass
class ScalarElementKinds:
    paths: set[Path] = field(default_factory=set)
    stamps: set[dt.datetime] = field(default_factory=set)
    days: set[dt.date] = field(default_factory=set)
    clocks: set[dt.time] = field(default_factory=set)
    flags: set[bool] = field(default_factory=set)
    ratios: set[float] = field(default_factory=set)
    colours: set[Colour] = field(default_factory=set)
    levels: set[Level] = field(default_factory=set)
    modes: set[Mode] = field(default_factory=set)


def test_every_supported_scalar_is_admitted_as_a_set_element():
    """A supported scalar hashes by a base's implementation as often as its own.

    ``Path`` hashes by ``PurePath`` and ``IntEnum`` by ``int``, so reading only the
    annotated class would reject both.
    """
    built = from_dict(
        ScalarElementKinds,
        {
            "paths": ["/tmp/a"],
            "stamps": ["2020-01-02T03:04:05"],
            "days": ["2020-01-02"],
            "clocks": ["03:04:05"],
            "flags": [True],
            "ratios": [1.5],
            "colours": ["red"],
            "levels": [1],
            "modes": ["fast"],
        },
    )
    assert built.paths == {Path("/tmp/a")}
    assert built.stamps == {dt.datetime(2020, 1, 2, 3, 4, 5)}
    assert built.levels == {Level.LOW}
    assert from_dict(ScalarElementKinds, to_dict(built)) == built


# --- a trial walk keeps nested findings and reports each once ----------------


@dataclass
class NestedUnsupportedTupleSet:
    values: set[tuple[deque[int], int]] = field(default_factory=set)


@dataclass
class NestedUnsupportedUnionSet:
    values: set[int | deque[int]] = field(default_factory=set)


@dataclass
class RepeatedInner:
    x: int = "wrong"  # pyrefly: ignore[bad-assignment]


@dataclass
class RepeatedSectionTupleSet:
    values: set[tuple[RepeatedInner, RepeatedInner]] = field(default_factory=set)


@pytest.mark.parametrize(
    "config_cls",
    [NestedUnsupportedTupleSet, NestedUnsupportedUnionSet],
    ids=["nested-in-tuple", "nested-in-union"],
)
def test_an_unsupported_annotation_nested_in_an_element_is_the_whole_finding(config_cls: type[Any]):
    # What the element rebuilds is undefined once something inside it has no
    # construction semantics, so the boundary message stands alone.
    messages = [message for _path, message in _issues(config_cls)]
    assert len(messages) == 1, messages
    assert messages[0].startswith("unsupported field type deque[int]"), messages


def test_a_set_element_reports_what_the_same_annotation_reports_as_a_field():
    """A trial walk carries the findings an ordinary field of that shape carries.

    ``tuple[A, B]`` reports one authored-default issue per declaration, so two
    classes that render alike stay two findings and fixing one leaves the other
    visible.
    """
    reported = _issues(RepeatedSectionTupleSet)
    assert sum(1 for _path, message in reported if "cannot be built" in message) == 1, reported
    defaults = [(path, message) for path, message in reported if "invalid authored default" in message]
    assert len(defaults) == 2, defaults
    assert {path for path, _message in defaults} == {"values.x"}


# --- a supplied subclass carries its own hash into construction ---------------


class UnhashableStr(str):
    __hash__: ClassVar[None] = None


class RaisingDate(dt.date):
    def __hash__(self) -> int:
        """Fail, so a construction that hashes it is visible.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("date hash ran")


@dataclass
class SuppliedStrings:
    values: set[str] = field(default_factory=set)


@dataclass
class SuppliedDates:
    values: set[dt.date] = field(default_factory=set)


@pytest.mark.parametrize(
    ("config_cls", "supplied", "rendered"),
    [
        (SuppliedStrings, UnhashableStr("x"), "set[str]"),
        (SuppliedDates, RaisingDate(2020, 1, 2), "set[date]"),
    ],
    ids=["suppressed-hash", "raising-hash"],
)
def test_a_supplied_subclass_carrying_its_own_hash_reports_as_an_issue(
    config_cls: type[Any], supplied: Any, rendered: str
):
    """Coercion keeps a value that already satisfies its annotation.

    A subclass instance handed straight to ``from_dict`` therefore reaches the set
    with a hash of its own, and whatever that hash does arrives under the
    annotation as written rather than past the collector.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {"values": [supplied]})
    messages = [issue.message for issue in info.value.issues]
    assert any(f"cannot build {rendered}" in message for message in messages), messages


# --- a failing hash is reported, an interpreter failure is not ----------------


class MemoryHungryStr(str):
    def __hash__(self) -> int:
        """Fail the way the interpreter reports on itself.

        Raises:
          MemoryError: Always.
        """
        raise MemoryError("out")


class UnprintableError(Exception):
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __str__(self) -> str:
        """Fail while being rendered.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("exception __str__ ran")


class UnprintableHashStr(str):
    def __hash__(self) -> int:
        """Fail with an exception that cannot be rendered.

        Raises:
          UnprintableError: Always.
        """
        raise UnprintableError


def test_an_interpreter_resource_failure_travels_to_the_caller():
    # A MemoryError describes the process rather than the config, so naming it an
    # invalid annotation would report the wrong cause.
    with pytest.raises(MemoryError):
        from_dict(SuppliedStrings, {"values": [MemoryHungryStr("x")]})


def test_an_exception_that_fails_to_render_is_still_reported():
    # Rendering a caught exception is another call into user code, so the class
    # name answers for one that raises.
    with pytest.raises(ConfigError) as info:
        from_dict(SuppliedStrings, {"values": [UnprintableHashStr("x")]})
    messages = [issue.message for issue in info.value.issues]
    assert any("UnprintableError" in message for message in messages), messages


# --- a nested invalid annotation names itself --------------------------------


class FloatValuedEnum(Enum):
    A = 1.5


@dataclass
class NestedLiteralSet:
    values: set[tuple[list[int], Literal[1.5]]] = field(default_factory=set)  # pyrefly: ignore[invalid-literal]


@dataclass
class NestedMappingKeySet:
    values: set[tuple[dict[int, str], int]] = field(default_factory=set)


@dataclass
class NestedEnumSet:
    values: set[list[FloatValuedEnum]] = field(default_factory=set)


@pytest.mark.parametrize(
    ("config_cls", "expected"),
    [
        (NestedLiteralSet, "Literal values must be primitive"),
        (NestedMappingKeySet, "unsupported dict key type int"),
        (NestedEnumSet, "must carry primitive values"),
    ],
    ids=["literal", "mapping-key", "enum-value"],
)
def test_an_invalid_annotation_nested_in_an_element_names_itself(config_cls: type[Any], expected: str):
    """The element's own schema says what is wrong with it, and it is reported.

    Each of these elements also holds a mutable container, which settles the
    container's instability on its own, so both statements are true and both are
    made.
    """
    messages = [message for _path, message in _issues(config_cls)]
    assert any(expected in message for message in messages), messages
    assert any("cannot be built" in message for message in messages), messages


@dataclass
class UndecidedTupleSet:
    values: set[tuple[int, deque[int]]] = field(default_factory=set)


@pytest.mark.parametrize(
    "config_cls",
    [UnsupportedGenericElementSet, NestedUnsupportedTupleSet, UndecidedTupleSet],
    ids=["direct", "nested-beside-list", "nested-beside-scalar"],
)
def test_an_element_whose_meaning_is_undefined_carries_no_instability_claim(config_cls: type[Any]):
    """A shape holding only scalars beside an unsupported annotation settles nothing.

    What ``deque[int]`` rebuilds is undefined, so the container has no ground to
    describe itself as unstable and reports the boundary message alone.
    """
    messages = [message for _path, message in _issues(config_cls)]
    assert len(messages) == 1, messages
    assert messages[0].startswith("unsupported field type deque[int]"), messages


# --- a container's own instability stands beside a nested finding -------------


@dataclass
class ListBesideBadMappingKey:
    values: set[tuple[list[int], dict[int, str]]] = field(default_factory=set)


@dataclass
class NestedAnySet:
    values: set[set[ConfigValue]] = field(default_factory=set)


@dataclass
class UnstableUnionSet:
    values: set[list[int] | dict[int, str]] = field(default_factory=set)


@pytest.mark.parametrize(
    "config_cls",
    [ListBesideBadMappingKey, NestedAnySet, UnstableUnionSet],
    ids=["list-beside-bad-key", "nested-any-set", "unstable-union"],
)
def test_an_independently_established_instability_survives_a_nested_finding(config_cls: type[Any]):
    """A mutable container settles the outer answer whatever sits beside it.

    Correcting only the nested defect would otherwise reveal a second rejection
    that was knowable from the start.
    """
    messages = [message for _path, message in _issues(config_cls)]
    assert len(messages) == 2, messages
    assert any("cannot be built: a set element must rebuild hashable" in message for message in messages), messages


# --- malformed specializations are a schema matter ---------------------------


@dataclass
class EllipsisOnlyTupleSet:
    values: set[tuple[...]] = field(default_factory=set)  # pyrefly: ignore[invalid-argument]


@dataclass
class EllipsisMidTupleSet:
    values: set[tuple[int, ..., str]] = field(default_factory=set)  # pyrefly: ignore[invalid-argument]


@dataclass
class EllipsisElementSet:
    values: set[...] = field(default_factory=set)  # pyrefly: ignore[invalid-param-spec]


@dataclass
class TwoArgumentSet:
    values: set[int, str] = field(default_factory=set)  # pyrefly: ignore[bad-specialization]


@pytest.mark.parametrize(
    ("config_cls", "expected"),
    [
        (EllipsisOnlyTupleSet, "carries ... outside the variadic form"),
        (EllipsisMidTupleSet, "carries ... outside the variadic form"),
        (EllipsisElementSet, "carries ..., which marks a variadic tuple"),
        (TwoArgumentSet, "builds from 1 type argument"),
    ],
    ids=["ellipsis-only-tuple", "ellipsis-mid-tuple", "ellipsis-element", "two-arguments"],
)
def test_a_malformed_specialization_is_reported_at_preflight(config_cls: type[Any], expected: str):
    """Python accepts these spellings at runtime, so preflight is what names them."""
    messages = [message for _path, message in _issues(config_cls)]
    assert any(expected in message for message in messages), messages


@dataclass
class WellFormedSpecializations:
    variadic: tuple[int, ...] = ()
    empty: tuple[()] = ()
    fixed: tuple[int, str] = (0, "")
    mapping: dict[str, int] = field(default_factory=dict)


def test_well_formed_specializations_still_build():
    built = from_dict(
        WellFormedSpecializations,
        {"variadic": [1, 2], "empty": [], "fixed": [1, "a"], "mapping": {"a": 1}},
    )
    assert built.variadic == (1, 2)
    assert built.empty == ()
    assert built.fixed == (1, "a")


# --- a failing render, and the interpreter's own state ------------------------


class NameRaisingMeta(type):
    def __getattribute__(cls, name: str) -> Any:
        """Fail while the class name is read.

        Args:
          name (str): The attribute being read.

        Returns:
          Any: The attribute, for every name other than ``__name__``.

        Raises:
          RuntimeError: When ``__name__`` is read.
        """
        if name == "__name__":
            raise RuntimeError("name read")
        return type.__getattribute__(cls, name)


class UnnameableError(Exception, metaclass=NameRaisingMeta):
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __str__(self) -> str:
        """Fail while being rendered.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("str ran")


class UnnameableHashStr(str):
    def __hash__(self) -> int:
        """Fail with an exception whose class name cannot be read.

        Raises:
          UnnameableError: Always.
        """
        raise UnnameableError


class RecursingEqStr(str):
    def __hash__(self) -> int:
        """Collide with every other value.

        Returns:
          int: A fixed hash.
        """
        return 0

    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __eq__(self, other: object) -> bool:
        """Exhaust the stack while comparing.

        Args:
          other (object): The value compared against.

        Returns:
          bool: Never returns.

        Raises:
          RecursionError: Always.
        """

        def deeper(depth: int) -> bool:
            return deeper(depth + 1)

        return deeper(0)


def test_an_exception_whose_name_cannot_be_read_is_still_described():
    # Reading the class name is the fallback, and it is user code too, so a fixed
    # phrase answers when it raises.
    with pytest.raises(ConfigError) as info:
        from_dict(SuppliedStrings, {"values": [UnnameableHashStr("x")]})
    messages = [issue.message for issue in info.value.issues]
    assert any("could not be described" in message for message in messages), messages


def test_a_recursion_failure_from_comparing_values_is_collected():
    # The depth belongs to the values being built rather than to the interpreter,
    # so it is reported as the construction failure it is.
    with pytest.raises(ConfigError) as info:
        from_dict(SuppliedStrings, {"values": [RecursingEqStr("a"), RecursingEqStr("b")]})
    messages = [issue.message for issue in info.value.issues]
    assert any("cannot build set[str]" in message for message in messages), messages


# --- a frozenset that builds is hashable --------------------------------------


@dataclass
class AnyFrozensetSet:
    values: set[frozenset[ConfigValue]] = field(default_factory=set)


@dataclass
class ListFrozensetSet:
    values: set[frozenset[list[int]]] = field(default_factory=set)


@dataclass
class FrozensetInTupleSet:
    values: set[tuple[frozenset[list[int]], int]] = field(default_factory=set)


@pytest.mark.parametrize(
    ("config_cls", "rendered"),
    [
        (AnyFrozensetSet, "frozenset[ConfigValue]"),
        (ListFrozensetSet, "frozenset[list[int]]"),
        (FrozensetInTupleSet, "frozenset[list[int]]"),
    ],
    ids=["any-member", "list-member", "inside-a-tuple"],
)
def test_a_frozenset_names_its_own_member_without_claiming_the_container(config_cls: type[Any], rendered: str):
    """Building a frozenset hashes its members, so a bad member stops it there.

    No load produces an unhashable frozenset, so the enclosing container has
    nothing of its own to report and the member's annotation is the finding.
    """
    messages = [message for _path, message in _issues(config_cls)]
    assert len(messages) == 1, messages
    assert f"{rendered} cannot be built" in messages[0], messages


# --- an enum's own type answers for it ----------------------------------------


class CertifyingEnumMeta(EnumType):
    # pyrefly: ignore[missing-override-decorator]
    def __instancecheck__(cls, instance: object) -> bool:
        """Certify anything as a member, leaving the member lookup as it stands.

        Args:
          instance (object): The object checked.

        Returns:
          bool: True, always.
        """
        return True


class LyingMode(Enum, metaclass=CertifyingEnumMeta):
    A = "a"

    @override
    @classmethod
    def _missing_(cls, value: object) -> Any:
        """Hand back a foreign object for a value no member carries.

        A lookup reaches this only after the member values miss, and the
        certifying metaclass above is what lets the foreign object past the
        standard lookup's own check on what this returns.

        Args:
          value (object): The value looked up.

        Returns:
          Any: A list, which no enum lookup should produce.
        """
        foreign: list[Any] = []
        return foreign


@dataclass
class LyingEnumSet:
    values: set[LyingMode] = field(default_factory=set)


def test_an_enum_result_is_confirmed_by_its_own_type():
    # The class being confirmed owns the hook an isinstance check would ask, so
    # the result's own type answers instead and the foreign value is reported at
    # the element's path. "zz" is a value no member carries, which is what routes
    # the lookup through _missing_ to reach the foreign object.
    with pytest.raises(ConfigError) as info:
        from_dict(LyingEnumSet, {"values": ["zz"]})
    assert [issue.path for issue in info.value.issues] == ["values.0"]


class RedirectingEnumMeta(EnumType):
    def __call__(cls, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Answer every lookup with the same member.

        Args:
          value (Any): The value looked up.
          *args (Any): Ignored.
          **kwargs (Any): Ignored.

        Returns:
          Any: The first member, whatever was asked for.
        """
        return next(iter(cls))


class RedirectedMode(Enum, metaclass=RedirectingEnumMeta):
    A = "a"
    B = "b"


@dataclass
class RedirectedEnumSet:
    values: set[RedirectedMode] = field(default_factory=set)


@dataclass
class RedirectedEnumScalar:
    value: RedirectedMode = RedirectedMode.B


@pytest.mark.parametrize(
    ("config_cls", "path"),
    [(RedirectedEnumSet, "values"), (RedirectedEnumScalar, "value")],
    ids=["set-element", "scalar-field"],
)
def test_an_enum_binding_its_own_lookup_is_reported_at_preflight(config_cls: type[Any], path: str):
    # Every member writes its own value, so a lookup that answers with one member
    # rebuilds B's value as A. The annotation says so before any value is read,
    # and it says so wherever the enum is named rather than in sets alone.
    messages = dict(_issues(config_cls))
    assert "looks a member up through RedirectingEnumMeta" in messages[path], messages
    assert messages[path].endswith("after a member's own value resolves"), messages


class Shade(Enum):
    RED = "red"


class Rank(IntEnum):
    LOW = 1


class Speed(StrEnum):
    FAST = "fast"


class Aliased(Enum):
    A = "a"
    B = "a"  # noqa: PIE796  (an alias is the behavior under test)


class Fallback(Enum):
    A = "a"

    @classmethod
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def _missing_(cls, value: object) -> Fallback:
        """Resolve every unknown value to one member.

        Args:
          value (object): The value that matched no member.

        Returns:
          Fallback: The single member.
        """
        return cls.A


@pytest.mark.parametrize(
    ("enum_cls", "supplied", "expected_name"),
    [
        (Shade, "red", "RED"),
        (Rank, 1, "LOW"),
        (Speed, "fast", "FAST"),
        (Aliased, "a", "A"),
        (Fallback, "unknown", "A"),
    ],
    ids=["enum", "int-enum", "str-enum", "alias", "missing-hook"],
)
def test_every_ordinary_enum_form_still_resolves(enum_cls: type[Enum], supplied: Any, expected_name: str):
    holder = make_dataclass("EnumHolder", [("values", set[enum_cls], field(default_factory=set))])
    built = from_dict(holder, {"values": [supplied]})
    assert {member.name for member in built.values} == {expected_name}


# --- reporting one failure stays one failure ----------------------------------


class FormatRaisingStr(str):
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __format__(self, format_spec: str) -> str:
        """Fail while being formatted.

        Args:
          format_spec (str): The format specification.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("format ran")


class FormatRaisingError(Exception):
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __str__(self) -> str:
        """Render as a string carrying its own formatting hook.

        Returns:
          str: A str subclass whose formatting raises.
        """
        return FormatRaisingStr("bad")


class FormatBombHashStr(str):
    def __hash__(self) -> int:
        """Fail with an exception whose text cannot be formatted.

        Raises:
          FormatRaisingError: Always.
        """
        raise FormatRaisingError


class ResourceRaisingError(Exception):
    # pyrefly: ignore[missing-override-decorator]  (typing.override needs 3.12; the floor is 3.11)
    def __str__(self) -> str:
        """Fail the way the interpreter reports on itself.

        Raises:
          MemoryError: Always.
        """
        raise MemoryError("render allocation")


class ResourceRenderHashStr(str):
    def __hash__(self) -> int:
        """Fail with an exception whose rendering exhausts memory.

        Raises:
          ResourceRaisingError: Always.
        """
        raise ResourceRaisingError


def test_text_that_cannot_be_formatted_is_copied_before_it_is_reported():
    with pytest.raises(ConfigError) as info:
        from_dict(SuppliedStrings, {"values": [FormatBombHashStr("x")]})
    messages = [issue.message for issue in info.value.issues]
    assert any("cannot build set[str]: bad" in message for message in messages), messages


def test_a_resource_failure_while_rendering_still_travels_to_the_caller():
    with pytest.raises(MemoryError):
        from_dict(SuppliedStrings, {"values": [ResourceRenderHashStr("x")]})
