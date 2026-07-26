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
from typing import (
    Any,
    NamedTuple,
    TypedDict,
)

import pytest

from confingo import (
    ConfigError,
    from_dict,
    to_dict,
)


BOUNDARY_REMEDY = (
    "choose a supported annotation (bool, int, float, str, Path, date/time, Enum/Literal, dataclass, "
    "container/union, array/tensor, or Any) and derive other runtime values in an init=False field"
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
    x: Any = None


def test_to_dict_rejects_unserializable_value():
    cfg = AnyField(x=Decimal("1.5"))
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
