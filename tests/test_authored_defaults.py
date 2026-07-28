"""Tests for authored defaults: validated against annotation and plain form, never coerced.

A direct ``field(default=...)`` is checked during schema preflight, so a wrong
default reports whether or not the input supplies the field. A ``default_factory``
runs once at the build that selects it, and its one product goes through the same
validation before reaching the object.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import UserDict
from collections.abc import Mapping  # noqa: TC003  (needed at runtime by get_type_hints)
from dataclasses import (
    dataclass,
    field,
)
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Literal,
)

import pytest
import yaml

from confingo import (
    ConfigError,
    ConfigScalar,
    ConfigValue,
)
from confingo.functional import (
    dumps_json,
    dumps_yaml,
    from_dict,
    to_dict,
)


np = pytest.importorskip("numpy")
npt = pytest.importorskip("numpy.typing")


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class Optimizer:
    name: str = "adam"
    lr: float = 1e-3


def _issues(config_cls: type[Any], data: dict[str, Any] | None = None) -> dict[str, str]:
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


# --- direct defaults fail at static preflight ---------------------------------


@dataclass
class WrongScalar:
    ratio: float = 1


@dataclass
class WrongPath:
    output_dir: Path = "runs"  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongInt:
    count: int = 3.0  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongTupleMember:
    tags: tuple[str, ...] = ("a", 2)  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongTupleArity:
    span: tuple[int, int] = (1, 2, 3)  # pyrefly: ignore[bad-assignment]


@dataclass
class MappingForSection:
    # A mapping cannot be written as a direct default at all, so the section-shaped
    # mistake reaches confingo through a factory.
    opt: Optimizer = field(default_factory=lambda: {"lr": 0.1})  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongUnion:
    limit: int | None = "none"  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongEnum:
    mode: Mode = "fast"  # pyrefly: ignore[bad-assignment]


@dataclass
class WrongLiteral:
    kind: Literal["a", "b"] = "c"  # pyrefly: ignore[bad-assignment]


@dataclass
class DatetimeOnDate:
    when: dt.date = dt.datetime(2026, 1, 1, 12)


@dataclass
class NonFinite:
    throughput: float = float("inf")


@dataclass
class OpaqueAny:
    # Decimal sits outside ConfigValue, which is what the preflight message below
    # reports; the schema is ill-typed on purpose.
    value: ConfigValue = Decimal("1.5")  # pyrefly: ignore[bad-assignment]


def test_wrong_scalar_default_names_the_annotation_and_the_rule():
    assert _issues(WrongScalar) == {
        "ratio": (
            "invalid authored default: expected a value already matching float, got int; "
            "defaults are validated as written"
        )
    }


def test_path_default_written_as_a_string_is_rejected():
    assert _issues(WrongPath)["output_dir"] == (
        "invalid authored default: expected a value already matching Path, got str; defaults are validated as written"
    )


def test_non_finite_float_default_is_rejected():
    assert _issues(NonFinite) == {"throughput": "invalid authored default: expected a finite float, got inf"}


def test_open_data_default_outside_the_domain_is_rejected():
    assert _issues(OpaqueAny) == {
        "value": (
            "invalid authored default: expected plain data for ConfigValue, got Decimal; "
            "use a scalar, a list, or a str-keyed mapping, or name the type with a dataclass section"
        )
    }


@pytest.mark.parametrize(
    ("config_cls", "path", "fragment"),
    [
        (WrongInt, "count", "already matching int, got float"),
        (WrongTupleMember, "tags.1", "already matching str, got int"),
        (WrongTupleArity, "span", "expected 2 items for tuple[int, int], got 3"),
        (MappingForSection, "opt", "already matching Optimizer, got dict"),
        (WrongUnion, "limit", "already matching int | None, got str"),
        (WrongEnum, "mode", "already matching Mode, got str"),
        (WrongLiteral, "kind", "expected one of 'a' | 'b', got 'c'"),
        (DatetimeOnDate, "when", "already matching date, got datetime"),
    ],
)
def test_defaults_outside_their_annotation_are_rejected(config_cls: type[Any], path: str, fragment: str):
    assert fragment in _issues(config_cls)[path]


def test_a_supplied_override_does_not_suppress_an_invalid_default():
    reported = _issues(WrongPath, {"output_dir": "elsewhere"})
    assert "invalid authored default" in reported["output_dir"]


@dataclass
class WrongDtype:
    a: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class NonFiniteArray:
    a: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf], dtype=np.float64))


@dataclass
class WrongNdim:
    a: np.ndarray[tuple[int, int], np.dtype[np.float64]] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class CorrectArray:
    a: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def test_array_dtype_shape_and_finiteness_are_checked():
    assert "defaults are validated as written" in _issues(WrongDtype)["a"]
    assert _issues(NonFiniteArray)["a.0"] == "invalid default_factory value: expected a finite float, got inf"
    assert "expected a 2-dimensional array" in _issues(WrongNdim)["a"]


def test_a_conforming_array_factory_is_accepted_unchanged():
    built = from_dict(CorrectArray, {})
    assert built.a.dtype == np.float64
    assert built.a.shape == (3,)


# --- correct defaults are kept exactly as written -----------------------------


@dataclass
class Correct:
    output_dir: Path = Path("runs")
    ratio: float = 1.0
    count: int = 3
    tags: tuple[str, ...] = ("a", "b")
    mode: Mode = Mode.FAST
    kind: Literal["a", "b"] = "a"
    when: dt.date = dt.date(2026, 1, 1)
    at: dt.time = dt.time(12, 0)
    stamp: dt.datetime = dt.datetime(2026, 1, 1, 12)
    limit: int | None = None


def test_correct_defaults_keep_object_identity():
    built = from_dict(Correct, {})
    assert built.output_dir is Correct.output_dir
    assert built.tags is Correct.tags
    assert built.mode is Mode.FAST


def test_a_config_built_entirely_from_defaults_round_trips_through_every_format():
    built = from_dict(Correct, {})
    exported = to_dict(built)
    assert from_dict(Correct, exported) == built
    assert from_dict(Correct, json.loads(dumps_json(built))) == built
    assert from_dict(Correct, yaml.safe_load(dumps_yaml(built))) == built


# --- factories run once, at the build that selects them -----------------------


class _Counter:
    """A default_factory recording how many times it was called.

    Attributes:
      calls (int): The number of invocations so far.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> list[int]:
        """Produce a fresh list and record the call.

        Returns:
          list[int]: A new two-element list on every call.
        """
        self.calls += 1
        return [1, 2]


COUNTER = _Counter()


@dataclass
class Counted:
    items: list[int] = field(default_factory=COUNTER)


def test_factory_runs_once_per_omitted_build_and_returns_a_fresh_object():
    before = COUNTER.calls
    first = from_dict(Counted, {})
    second = from_dict(Counted, {})
    assert COUNTER.calls == before + 2
    assert first.items == [1, 2]
    assert first.items is not second.items


def test_factory_is_not_called_for_a_supplied_field():
    before = COUNTER.calls
    built = from_dict(Counted, {"items": [9]})
    assert built.items == [9]
    assert COUNTER.calls == before


def _raising() -> list[int]:
    """Fail the way a factory reading a missing file would.

    Raises:
      ValueError: Always.
    """
    raise ValueError("cannot read seeds file")


@dataclass
class RaisingFactory:
    items: list[int] = field(default_factory=_raising)


@dataclass
class BadFactories:
    opt: Optimizer = field(default_factory=lambda: Optimizer(lr="fast"))  # pyrefly: ignore[bad-argument-type]
    mapping: dict[str, int] = field(default_factory=lambda: {1: "x"})  # pyrefly: ignore[bad-assignment]
    seeds: list[int] = field(default_factory=lambda: [1, "two"])  # pyrefly: ignore[bad-assignment]
    supplied: int = 0


def test_a_raising_factory_reports_at_the_field_path():
    assert _issues(RaisingFactory) == {"items": "default_factory raised ValueError: cannot read seeds file"}


def test_factory_and_input_issues_aggregate():
    reported = _issues(BadFactories, {"supplied": "nope"})
    assert reported["opt.lr"] == (
        "invalid default_factory value: expected a value already matching float, got str; "
        "defaults are validated as written"
    )
    assert reported["mapping.1"] == "invalid default_factory value: expected a str mapping key, got int"
    assert "already matching int, got str" in reported["seeds.1"]
    assert reported["supplied"] == "expected int, got str"


@dataclass
class BaselineSection:
    opt: Optimizer = field(default_factory=lambda: Optimizer(lr=1e-2))


def test_a_section_factory_builds_and_round_trips():
    built = from_dict(BaselineSection, {})
    assert built.opt == Optimizer(lr=1e-2)
    assert from_dict(BaselineSection, to_dict(built)) == built


# --- nested schema paths ------------------------------------------------------


@dataclass
class InnerWithBadDefault:
    lr: float = 1


@dataclass
class MiddleSection:
    inner: InnerWithBadDefault = field(default_factory=InnerWithBadDefault)


@dataclass
class OuterRoot:
    middle: MiddleSection = field(default_factory=MiddleSection)


def test_a_nested_direct_default_reports_at_its_full_schema_path():
    assert "middle.inner.lr" in _issues(OuterRoot)


@dataclass
class SiblingOfBadNested:
    inner: InnerWithBadDefault = field(default_factory=InnerWithBadDefault)
    own: Path = "runs"  # pyrefly: ignore[bad-assignment]


def test_every_invalid_direct_default_in_the_tree_reports_in_one_pass():
    # A nested class's own bad default is a value problem, not a structural one,
    # so it leaves the enclosing class's default still judged.
    reported = _issues(SiblingOfBadNested)
    assert "already matching float, got int" in reported["inner.lr"]
    assert "already matching Path, got str" in reported["own"]


@dataclass
class UnsupportedAnnotation:
    amount: Decimal = Decimal("1.5")


def test_an_unsupported_annotation_suppresses_its_own_default_issue():
    # The default can only restate the annotation problem, so the boundary
    # message stands alone.
    reported = _issues(UnsupportedAnnotation)
    assert "unsupported field type Decimal" in reported["amount"]
    assert "invalid authored default" not in reported["amount"]


# --- sections are held through a factory --------------------------------------


@dataclass(frozen=True)
class FrozenSection:
    x: int = 1


@dataclass
class HoldsFrozenSection:
    section: FrozenSection = field(default_factory=FrozenSection)


def test_a_frozen_section_default_factory_builds():
    built = from_dict(HoldsFrozenSection, {})
    assert built.section == FrozenSection(x=1)


def test_a_touched_section_directs_a_later_direct_default_to_the_factory():
    # A config class is unhashable, which is what @dataclass reads as a mutable
    # default, so the decorator names default_factory as the way to hold one.
    from_dict(HoldsFrozenSection, {})
    with pytest.raises(ValueError, match="use default_factory"):

        @dataclass
        class LaterDeclared:
            section: FrozenSection = FrozenSection()


# --- init=False keeps its exemption -------------------------------------------


@dataclass
class RuntimeState:
    lr: float = 0.1
    handle: Decimal = field(init=False, default=Decimal("1.5"))


def test_init_false_defaults_keep_the_type_boundary_exemption():
    built = from_dict(RuntimeState, {})
    assert built.handle == Decimal("1.5")


# --- set elements have to come back hashable ----------------------------------


@dataclass
class HashableSets:
    scalars: frozenset[ConfigScalar] = frozenset({1, "a"})
    typed: set[tuple[int, int]] = field(default_factory=lambda: {(1, 2)})


def test_annotated_and_scalar_set_elements_round_trip_through_every_format():
    built = from_dict(HashableSets, {})
    assert from_dict(HashableSets, to_dict(built)) == built
    assert from_dict(HashableSets, json.loads(dumps_json(built))) == built
    assert from_dict(HashableSets, yaml.safe_load(dumps_yaml(built))) == built


# --- mappings arrive as the dict construction builds --------------------------


@dataclass
class ProxyMapping:
    values: dict[str, int] = field(default_factory=lambda: MappingProxyType({"a": 1}))  # pyrefly: ignore[bad-assignment]


@dataclass
class UserDictMapping:
    values: Mapping[str, int] = field(default_factory=lambda: UserDict({"a": 1}))


@pytest.mark.parametrize("config_cls", [ProxyMapping, UserDictMapping])
def test_a_mapping_default_that_is_not_a_dict_is_rejected(config_cls: type[Any]):
    assert "defaults are validated as written" in _issues(config_cls)["values"]


# --- validating a default runs none of the schema's own code ------------------


SIDE_EFFECTS: list[str] = []


class FirstEnum(Enum):
    A = "a"

    @classmethod
    # pyrefly: ignore[missing-override-decorator]
    def _missing_(cls, value: object) -> None:
        """Record a lookup miss, so a validation pass that calls it is visible.

        Args:
          value (object): The value that matched no member.

        Returns:
          None: Always, leaving the miss unresolved.
        """
        SIDE_EFFECTS.append(f"missing:{value}")


class SecondEnum(Enum):
    B = "b"


@dataclass
class EnumUnionTuple:
    values: tuple[FirstEnum | SecondEnum, ...] = (SecondEnum.B,)


def _counted_factory() -> int:
    """Produce a value and record that the factory ran.

    Returns:
      int: A fixed value.
    """
    SIDE_EFFECTS.append("factory")
    return 7


@dataclass
class HookedSection:
    value: int = field(default_factory=_counted_factory)

    def __post_init__(self) -> None:
        SIDE_EFFECTS.append("post_init")

    def __validate__(self) -> list[str]:
        """Record that the hook ran.

        Returns:
          list[str]: No problems.
        """
        SIDE_EFFECTS.append("validate")
        return []


@dataclass
class UnionShiftsToSection:
    values: set[list[HookedSection] | tuple[Any, ...]] = field(default_factory=lambda: {(1, 2)})  # pyrefly: ignore[bad-assignment]


def test_a_valid_enum_union_default_runs_no_user_code():
    # The authored member belongs to the second enum, so a validation pass built
    # on coercion would try the first enum against the exported "b" and call its
    # _missing_ hook.
    SIDE_EFFECTS.clear()
    built = from_dict(EnumUnionTuple, {})
    assert built.values == (SecondEnum.B,)
    assert SIDE_EFFECTS == []


def test_a_section_bearing_set_annotation_is_rejected_without_running_it():
    # The annotation decides the rejection on its own, so the authored value is
    # left untouched and the section's factory and hooks stay unentered.
    SIDE_EFFECTS.clear()
    assert "cannot be built" in _issues(UnionShiftsToSection)["values"]
    assert SIDE_EFFECTS == []
