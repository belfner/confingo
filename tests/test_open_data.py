"""The open-data surface: ``ConfigValue`` / ``ConfigScalar`` and the forms they replace.

Three things are pinned here. The withdrawn annotations -- ``Any`` and every
argument-free container spelling -- each report the parameterized form to write
instead. The plain-data domain itself is checked at both boundaries a value
crosses: inbound through ``from_dict`` and as an authored default or factory
product. And every walk over a value is bounded, so a structure that reaches
itself or nests past the budget arrives as a path-tagged ``ConfigError`` rather
than a raw ``RecursionError``.
"""

from __future__ import annotations

import collections.abc as cabc
import datetime as dt
import typing
from dataclasses import (
    dataclass,
    field,
    make_dataclass,
)
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    get_origin,
)

import pytest

from confingo import (
    ConfigError,
    ConfigScalar,
    ConfigValue,
)
from confingo.functional import (
    config_equal,
    config_hash,
    from_dict,
    to_dict,
)


def holder(hint: Any, default: Any = None) -> type:
    """Build a one-field dataclass carrying a real (non-string) annotation."""
    if default is None:
        return make_dataclass("Holder", [("x", hint)])
    return make_dataclass("Holder", [("x", hint, default)])


def issues_of(config_cls: type, data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Build a config and return every issue as a (path, message) pair."""
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {} if data is None else data)
    return [(issue.path, issue.message) for issue in info.value.issues]


# --- the withdrawn annotations ------------------------------------------------


def test_any_names_the_alias_that_replaced_it():
    assert issues_of(holder(Any)) == [
        (
            "x",
            "Any leaves the values it holds undescribed; annotate the field ConfigValue "
            "(from confingo) for plain data of any shape, or name the type the field holds",
        )
    ]


BARE_FORMS: list[tuple[str, Any, str]] = [
    ("list", list, "list[ConfigValue]"),
    ("tuple", tuple, "tuple[ConfigValue, ...]"),
    ("dict", dict, "dict[str, ConfigValue]"),
    ("set", set, "set[ConfigScalar]"),
    ("frozenset", frozenset, "frozenset[ConfigScalar]"),
    ("Sequence", cabc.Sequence, "Sequence[ConfigValue]"),
    ("Mapping", cabc.Mapping, "Mapping[str, ConfigValue]"),
    ("typing.List", typing.List, "list[ConfigValue]"),  # noqa: UP006
    ("typing.Tuple", typing.Tuple, "tuple[ConfigValue, ...]"),  # noqa: UP006
    ("typing.Dict", typing.Dict, "dict[str, ConfigValue]"),  # noqa: UP006
    ("typing.Set", typing.Set, "set[ConfigScalar]"),  # noqa: UP006
    ("typing.FrozenSet", typing.FrozenSet, "frozenset[ConfigScalar]"),  # noqa: UP006
    ("typing.Sequence", typing.Sequence, "Sequence[ConfigValue]"),
    ("typing.Mapping", typing.Mapping, "Mapping[str, ConfigValue]"),
]


@pytest.mark.parametrize(
    ("hint", "written"),
    [(hint, written) for _name, hint, written in BARE_FORMS],
    ids=[name for name, _hint, _written in BARE_FORMS],
)
def test_every_argument_free_container_spelling_names_its_parameterized_form(hint: Any, written: str):
    rendered = written.split("[", 1)[0]
    assert issues_of(holder(hint)) == [
        (
            "x",
            f"{rendered} carries no element type; write {written} for plain data of any shape, "
            f"or name the element type",
        )
    ]


# The empty argument list is what these spell, so the checker reports each one as
# the under-specialization it is; the assertion is that confingo reports it too.
EMPTY_SUBSCRIPTS: list[tuple[str, Any, str]] = [
    ("list", list[()], "list[ConfigValue]"),  # pyrefly: ignore[bad-specialization]
    ("dict", dict[()], "dict[str, ConfigValue]"),  # pyrefly: ignore[bad-specialization]
    ("set", set[()], "set[ConfigScalar]"),  # pyrefly: ignore[bad-specialization]
    ("frozenset", frozenset[()], "frozenset[ConfigScalar]"),  # pyrefly: ignore[bad-specialization]
    ("Sequence", cabc.Sequence[()], "Sequence[ConfigValue]"),  # pyrefly: ignore[bad-specialization]
    ("Mapping", cabc.Mapping[()], "Mapping[str, ConfigValue]"),  # pyrefly: ignore[bad-specialization]
]


@pytest.mark.parametrize(
    ("hint", "written"),
    [(hint, written) for _name, hint, written in EMPTY_SUBSCRIPTS],
    ids=[name for name, _hint, _written in EMPTY_SUBSCRIPTS],
)
def test_an_empty_subscript_names_the_same_parameterized_form(hint: Any, written: str):
    # PEP 585 accepts an empty argument list on every origin, and only a tuple
    # reads it as a shape, so every other container names no element type.
    rendered = written.split("[", 1)[0]
    assert issues_of(holder(hint)) == [
        (
            "x",
            f"{rendered} carries no element type; write {written} for plain data of any shape, "
            f"or name the element type",
        )
    ]


def test_the_explicit_empty_tuple_stays_distinct_from_the_bare_one():
    # ``tuple[()]`` names a tuple holding nothing, which is a shape confingo builds;
    # the argument-free spelling names no element type at all.
    built = from_dict(holder(tuple[()], field(default_factory=tuple)), {})
    assert built.x == ()
    assert to_dict(built) == {"x": []}


# --- the inbound plain-data domain --------------------------------------------


def test_config_value_accepts_plain_data_and_rebuilds_sequences_as_lists():
    built = from_dict(holder(ConfigValue), {"x": {"a": [1, 2.5, "t", True, None], "b": (3, 4)}})
    assert built.x == {"a": [1, 2.5, "t", True, None], "b": [3, 4]}
    assert to_dict(built) == {"x": {"a": [1, 2.5, "t", True, None], "b": [3, 4]}}


def test_config_scalar_admits_the_leaf_half_alone():
    assert from_dict(holder(ConfigScalar), {"x": "text"}).x == "text"
    assert from_dict(holder(ConfigScalar), {"x": None}).x is None
    assert issues_of(holder(ConfigScalar), {"x": [1]}) == [
        ("x", "expected one plain scalar for ConfigScalar, got list")
    ]


def test_a_value_outside_the_domain_names_the_shapes_that_are_inside_it():
    assert issues_of(holder(ConfigValue), {"x": Path("p")}) == [
        (
            "x",
            "expected plain data for ConfigValue, got PosixPath; use a scalar, a list, "
            "or a str-keyed mapping, or name the type with a dataclass section",
        )
    ]


def test_a_non_finite_float_has_no_plain_form():
    assert issues_of(holder(ConfigValue), {"x": float("inf")}) == [("x", "expected a finite float, got inf")]


def test_a_non_str_mapping_key_is_named_by_its_type():
    assert issues_of(holder(ConfigValue), {"x": {1: "v"}}) == [("x", "expected a str mapping key, got int")]


def test_a_key_whose_str_raises_is_reported_rather_than_run():
    class Raising:
        def __hash__(self) -> int:
            return 1

        def __str__(self) -> str:
            raise RuntimeError("__str__ called")

    assert issues_of(holder(ConfigValue), {"x": {Raising(): 1}}) == [("x", "expected a str mapping key, got Raising")]


def test_sibling_entries_still_report_together():
    found = issues_of(holder(ConfigValue), {"x": {"a": Path("p"), "b": float("nan")}})
    assert found == [
        (
            "x.a",
            "expected plain data for ConfigValue, got PosixPath; use a scalar, a list, "
            "or a str-keyed mapping, or name the type with a dataclass section",
        ),
        ("x.b", "expected a finite float, got nan"),
    ]


# --- authored defaults carry the same domain ----------------------------------


@dataclass
class ScalarDefault:
    # A Path is written as text by the marshal walk, which is a conversion a
    # default never gets; the schema is ill-typed on purpose.
    value: ConfigScalar = Path("not-a-scalar")  # pyrefly: ignore[bad-assignment]


@dataclass
class ScalarFactory:
    value: ConfigScalar = field(default_factory=list)  # pyrefly: ignore[bad-assignment]


@dataclass
class ValueDefault:
    value: ConfigValue = ()  # pyrefly: ignore[bad-assignment]


@dataclass
class ValueFactory:
    value: ConfigValue = field(default_factory=set)  # pyrefly: ignore[bad-assignment]


def nested_path_factory() -> ConfigValue:
    """Produce open data holding a Path, which the domain declines."""
    # A Path is written as text by the marshal walk, which is a conversion a
    # factory product never gets; the return is out of domain on purpose.
    return {"a": [Path("p")]}  # pyrefly: ignore[bad-return]


def plain_factory() -> ConfigValue:
    """Produce open data every part of which is a member of the domain.

    Annotating the factory is what a mutable open-data default is written as: the
    return type gives the literal an expected type, so a checker reads the nested
    literal against ``ConfigValue`` rather than against its own inferred type.
    """
    return {"a": [1, 2.5, None, True]}


@dataclass
class NestedValueFactory:
    value: ConfigValue = field(default_factory=nested_path_factory)


@dataclass
class PlainValueFactory:
    value: ConfigValue = field(default_factory=plain_factory)


@pytest.mark.parametrize(
    ("config_cls", "path", "message"),
    [
        (ScalarDefault, "value", "invalid authored default: expected one plain scalar for ConfigScalar, got PosixPath"),
        (ScalarFactory, "value", "invalid default_factory value: expected one plain scalar for ConfigScalar, got list"),
        (
            ValueDefault,
            "value",
            "invalid authored default: expected plain data for ConfigValue, got tuple; use a scalar, a list, "
            "or a str-keyed mapping, or name the type with a dataclass section",
        ),
        (
            ValueFactory,
            "value",
            "invalid default_factory value: expected plain data for ConfigValue, got set; use a scalar, a list, "
            "or a str-keyed mapping, or name the type with a dataclass section",
        ),
        (
            NestedValueFactory,
            "value.a.0",
            "invalid default_factory value: expected plain data for ConfigValue, got PosixPath; use a scalar, "
            "a list, or a str-keyed mapping, or name the type with a dataclass section",
        ),
    ],
    ids=["scalar-default", "scalar-factory", "value-default", "value-factory", "nested-factory"],
)
def test_an_authored_default_outside_the_domain_is_rejected(config_cls: type, path: str, message: str):
    assert issues_of(config_cls) == [(path, message)]


def test_a_default_inside_the_domain_is_passed_on_unchanged():
    assert from_dict(PlainValueFactory, {}).value == {"a": [1, 2.5, None, True]}


# --- every walk over a value is bounded ---------------------------------------


@dataclass
class Open:
    value: ConfigValue = None


@dataclass
class Section:
    child: Section | None = None


CYCLE_MESSAGE = "value holds itself, so it has no plain form; supply a structure that terminates"
DEPTH_MESSAGE = (
    "nesting reaches the 64 level limit for plain data; "
    "flatten the structure, or name the shape with a dataclass section"
)


def self_holding_list() -> list[Any]:
    """Build a list that holds itself."""
    items: list[Any] = []
    items.append(items)
    return items


def deep_mapping(levels: int) -> dict[str, Any]:
    """Build a chain of singly nested mappings."""
    root: dict[str, Any] = {}
    cursor = root
    for _ in range(levels):
        child: dict[str, Any] = {}
        cursor["child"] = child
        cursor = child
    return root


def test_a_supplied_cycle_is_reported_at_the_value_that_closes_it():
    assert issues_of(Open, {"value": self_holding_list()}) == [("value.0", CYCLE_MESSAGE)]


def test_a_supplied_structure_deeper_than_the_budget_is_reported():
    deep: Any = 1
    for _ in range(80):
        deep = [deep]
    assert issues_of(Open, {"value": deep})[0][1] == DEPTH_MESSAGE


def nested_lists(levels: int) -> Any:
    """Build a value of exactly this many nested lists around one scalar."""
    deep: Any = 1
    for _ in range(levels):
        deep = [deep]
    return deep


@pytest.mark.parametrize("levels", [62, 63], ids=["inside", "at-the-edge"])
def test_a_value_inside_the_budget_survives_every_walk(levels: int):
    payload = nested_lists(levels)

    def factory() -> ConfigValue:
        return payload

    @dataclass
    class Factory:
        value: ConfigValue = field(default_factory=factory)

    built = from_dict(Open, {"value": payload})
    assert to_dict(built) == {"value": payload}
    assert config_hash(built) == config_hash(built)
    assert config_equal(built, built)
    assert from_dict(Factory, {}).value == payload


@pytest.mark.parametrize("levels", [64, 65], ids=["at-the-limit", "past-it"])
def test_a_value_past_the_budget_is_declined_by_every_walk(levels: int):
    # One budget over the whole plain document, so a value the load declines is
    # also one no authored default or factory can smuggle past export.
    payload = nested_lists(levels)

    def factory() -> ConfigValue:
        return payload

    @dataclass
    class Factory:
        value: ConfigValue = field(default_factory=factory)

    assert issues_of(Open, {"value": payload})[0][1] == DEPTH_MESSAGE
    assert issues_of(Factory)[0][1] == f"invalid default_factory value: {DEPTH_MESSAGE}"


def test_a_cycle_through_a_recursive_section_is_reported():
    cyclic: dict[str, Any] = {}
    cyclic["child"] = cyclic
    assert issues_of(Section, cyclic) == [("child", CYCLE_MESSAGE)]


def test_a_recursive_section_deeper_than_the_budget_is_reported():
    assert issues_of(Section, deep_mapping(200))[0][1] == DEPTH_MESSAGE


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        ("to_dict", to_dict),
        ("config_hash", config_hash),
        ("config_equal", lambda config: config_equal(config, config)),
    ],
    ids=["to_dict", "config_hash", "config_equal"],
)
def test_a_cycle_reaching_an_operation_reports_instead_of_recursing(name: str, operation: Any):
    from_dict(Open, {})  # schema processing installs canonical equality
    held = Open(self_holding_list())
    with pytest.raises(ConfigError) as info:
        operation(held)
    assert any(issue.message == CYCLE_MESSAGE for issue in info.value.issues), name


def test_a_cyclic_factory_product_is_reported_rather_than_recursing():
    @dataclass
    class Cyclic:
        value: ConfigValue = field(default_factory=self_holding_list)

    assert issues_of(Cyclic) == [("value.0", f"invalid default_factory value: {CYCLE_MESSAGE}")]


# --- a set element names one type ----------------------------------------------


class PlainColor(Enum):
    """An enum whose values are text of its own."""

    RED = "red"


class Shade(Enum):
    """A second enum, whose values no other enum here carries."""

    DARK = "dark"


UNION_ELEMENTS: list[tuple[str, Any, str]] = [
    ("str-path", set[str | Path], "str | Path"),
    ("path-str", frozenset[Path | str], "Path | str"),
    ("int-float", set[int | float], "int | float"),
    ("int-str", set[int | str], "int | str"),
    ("date-time", set[dt.date | dt.time], "date | time"),
    ("enum-enum", set[PlainColor | Shade], "PlainColor | Shade"),
    ("literal-str", set[Literal["a"] | str], "'a' | str"),  # noqa: PYI051
    ("tuple-frozenset", set[tuple[int] | frozenset[str]], "tuple[int] | frozenset[str]"),
    ("optional-pair", set[str | Path | None], "str | Path | None"),
]


@pytest.mark.parametrize(
    ("hint", "union"),
    [(hint, union) for _name, hint, union in UNION_ELEMENTS],
    ids=[name for name, _hint, _union in UNION_ELEMENTS],
)
def test_a_set_element_naming_a_union_is_rejected(hint: Any, union: str):
    found = issues_of(holder(hint, field(default_factory=set)))
    assert len(found) == 1, found
    path, message = found[0]
    assert path == "x"
    assert f"a set element names one type, and {union} names several" in message
    assert "write T | None for an optional element" in message


def test_a_union_nested_in_an_element_shape_is_reached():
    found = issues_of(holder(set[tuple[int, str | Path]], field(default_factory=set)))
    assert len(found) == 1, found
    assert "str | Path names several" in found[0][1]


NAMED_ELEMENTS: list[tuple[str, Any, Any]] = [
    ("str", set[str], {"a", "b"}),
    ("optional-str", set[str | None], {"a", None}),
    ("optional-path", frozenset[Path | None], frozenset({Path("a"), None})),
    ("scalar", set[ConfigScalar], {1, "a", None}),
    ("enum", set[PlainColor], {PlainColor.RED}),
    ("literal", set[Literal["a", "b"]], {"a", "b"}),
    ("tuple", set[tuple[int, str]], {(1, "a"), (2, "b")}),
    ("optional-in-tuple", set[tuple[int, str | None]], {(1, "a"), (2, None)}),
    ("frozenset", set[frozenset[str]], {frozenset({"a"}), frozenset({"b"})}),
]


@pytest.mark.parametrize(
    ("hint", "elements"),
    [(hint, elements) for _name, hint, elements in NAMED_ELEMENTS],
    ids=[name for name, _hint, _elements in NAMED_ELEMENTS],
)
def test_every_admitted_set_element_round_trips_its_elements(hint: Any, elements: Any):
    # One named type writes one plain form per value and reads each back through
    # itself, and ``null`` is a form no other reader accepts, so the elements a
    # file carries are the elements a load rebuilds.
    factory = frozenset if get_origin(hint) is frozenset else set
    config_cls = holder(hint, field(default_factory=factory))
    config = config_cls(x=elements)
    plain = to_dict(config)
    rebuilt = from_dict(config_cls, plain)
    assert len(rebuilt.x) == len(elements)
    assert rebuilt == config
    assert config_hash(rebuilt) == config_hash(config)
    assert to_dict(rebuilt) == plain


def test_an_ordinary_union_field_is_untouched_by_the_set_rule():
    # The rule is about what a set can keep apart, so every other position still
    # takes a union and reads its members in declaration order.
    @dataclass
    class Mixed:
        value: int | str = 0
        items: list[str | Path] = field(default_factory=list)

    built = from_dict(Mixed, {"value": 1, "items": ["a"]})
    assert built.value == 1
    assert built.items == ["a"]
    assert from_dict(Mixed, to_dict(built)) == built
