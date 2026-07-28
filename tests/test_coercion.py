from __future__ import annotations

from dataclasses import dataclass
from enum import (
    Enum,
    EnumType,
)
from pathlib import Path
from typing import (
    Annotated,
    Any,
    override,
)

import pytest

from confingo import ConfigError
from confingo.functional import (
    config_equal,
    config_hash,
    from_dict,
    to_dict,
)
from tests.schemas import (
    Containers,
    Device,
    LiteralInts,
    Trainer,
    Training,
)


def test_integral_float_to_int():
    cfg = from_dict(Training, {"buffer_size": 2e6})
    assert cfg.buffer_size == 2_000_000
    assert isinstance(cfg.buffer_size, int)


def test_enum_by_value():
    assert from_dict(Training, {"device": "cuda"}).device is Device.CUDA


def test_enum_by_name():
    assert from_dict(Training, {"device": "CUDA"}).device is Device.CUDA


def test_str_to_path():
    assert from_dict(Training, {"output_dir": "runs/exp"}).output_dir == Path("runs/exp")


def test_bool_rejected_on_int():
    with pytest.raises(ConfigError) as info:
        from_dict(Training, {"buffer_size": True})
    assert any(issue.path == "buffer_size" for issue in info.value.issues)


def test_literal_membership():
    assert from_dict(Trainer, {"algorithm": "sac"}).algorithm == "sac"
    with pytest.raises(ConfigError):
        from_dict(Trainer, {"algorithm": "ppo"})


def test_literal_bool_int_distinction():
    assert from_dict(LiteralInts, {"level": 2}).level == 2
    # True == 1 but is a bool, so it fails a Literal[1, 2] of ints.
    with pytest.raises(ConfigError):
        from_dict(LiteralInts, {"level": True})
    with pytest.raises(ConfigError):
        from_dict(LiteralInts, {"level": 3})


def test_sequence_container_types():
    cfg = from_dict(Containers, {"names": ["a", "b"], "frozen": [1, 2]})
    assert cfg.names == {"a", "b"}
    assert isinstance(cfg.names, set)
    assert cfg.frozen == frozenset({1, 2})
    assert isinstance(cfg.frozen, frozenset)


def test_fixed_tuple_arity_mismatch():
    with pytest.raises(ConfigError):
        from_dict(Containers, {"pair": [1, 2, 3]})


def test_variadic_tuple():
    assert from_dict(Containers, {"variadic": [1, 2, 3, 4]}).variadic == (1, 2, 3, 4)


@dataclass
class IntFirst:
    x: int | float


@dataclass
class FloatFirst:
    x: float | int


@dataclass
class NumericDefault:
    x: int | float = 1.0


@pytest.mark.parametrize("config_cls", [IntFirst, FloatFirst], ids=["int-first", "float-first"])
@pytest.mark.parametrize("written", [1, 1.0], ids=["int-value", "float-value"])
def test_a_numeric_union_rebuilds_the_member_the_plain_form_carries(config_cls: type[Any], written: float):
    # An int field accepts an integral float so 1e6 is read, which would let
    # declaration order alone send a float to the int member and back. The member
    # naming the class the plain form carries goes first, so both orders hold.
    source = config_cls(written)
    rebuilt = from_dict(config_cls, to_dict(source))
    assert type(rebuilt.x) is type(written)
    assert config_equal(source, rebuilt)
    assert config_hash(source) == config_hash(rebuilt)


def test_a_numeric_union_default_reloads_as_the_type_it_was_authored_in():
    # The invariant covers a validated authored default, so the float this one
    # carries has to survive the export and the load that reads it back.
    source = from_dict(NumericDefault, {})
    rebuilt = from_dict(NumericDefault, to_dict(source))
    assert type(source.x) is float
    assert type(rebuilt.x) is float
    assert config_equal(source, rebuilt)


def test_the_integral_float_conversion_survives_outside_a_numeric_union():
    # Ordering by the carried class applies where two numeric members compete;
    # a lone int field still reads the 1e6 a file spells as a float.
    assert from_dict(Training, {"buffer_size": 2e6}).buffer_size == 2_000_000


@dataclass
class PathFirst:
    x: Path | str


@dataclass
class StrFirst:
    x: str | Path


@pytest.mark.parametrize(
    ("config_cls", "expected"),
    [(PathFirst, Path("a")), (StrFirst, "a")],
    ids=["path-first", "str-first"],
)
def test_declaration_order_still_decides_where_one_plain_form_fits_both(config_cls: type[Any], expected: object):
    # A Path and a str are both written as a string, so the plain form names
    # neither member on its own and the declared order is what answers.
    built = from_dict(config_cls, {"x": "a"})
    assert built.x == expected
    assert type(built.x) is type(expected)


class Number(Enum):
    ONE = 1


class Flagged(Enum):
    YES = True


@dataclass
class EnumBeforeInt:
    x: Number | int


@dataclass
class EnumBeforeAnnotatedInt:
    x: Number | Annotated[int, "tag"]


@dataclass
class EnumBeforeBool:
    x: Flagged | bool


@pytest.mark.parametrize(
    ("config_cls", "supplied", "expected"),
    [
        (EnumBeforeInt, 1, Number.ONE),
        (EnumBeforeAnnotatedInt, 1, Number.ONE),
        (EnumBeforeBool, True, Flagged.YES),
    ],
    ids=["enum-int", "enum-annotated-int", "enum-bool"],
)
def test_one_numeric_member_beside_another_kind_keeps_declaration_order(
    config_cls: type[Any], supplied: object, expected: object
):
    # Ordering by the carried class settles a pair of numeric members. One
    # numeric member beside a member of another kind is a union whose order the
    # author chose among kinds, so the declared member answers.
    assert from_dict(config_cls, {"x": supplied}).x is expected


class RaisingMissing(Enum):
    A = "a"

    @override
    @classmethod
    def _missing_(cls, value: object) -> None:
        """Raise for a value no member carries.

        Args:
          value (object): The value looked up.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("missing boom")


@dataclass
class HasRaisingMissing:
    x: RaisingMissing


def test_a_missing_hook_that_raises_is_reported_at_the_value_path():
    # _missing_ belongs to the config author and runs inside the lookup, so what
    # it raises describes the config and arrives as one issue at this path.
    with pytest.raises(ConfigError) as info:
        from_dict(HasRaisingMissing, {"x": "unknown"})
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert "resolving enum RaisingMissing raised RuntimeError: missing boom" in issue.message
    assert issue.message.endswith(
        "leave the member values, the member names, and any _missing_ hook answering for the values a file carries"
    )


class MembersTrap(EnumType):
    """A metaclass that keeps the standard lookup and raises on the member mapping."""

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__members__":
            raise RuntimeError("members lookup boom")
        return super().__getattribute__(name)


class Trapped(Enum, metaclass=MembersTrap):
    A = "a"

    @override
    @classmethod
    def _missing_(cls, value: object) -> None:
        """Report a value outside the member values by answering with None.

        Args:
          value (object): The value looked up.

        Returns:
          None: Always, which sends the lookup on to the name fallback.
        """


@dataclass
class HasTrapped:
    x: Trapped


def test_the_member_name_fallback_is_inside_the_same_containment():
    # The value lookup answers None here, so resolution goes on to read the
    # member mapping. That read belongs to the enum's author too, so the whole
    # resolution runs under one containment and reports at this path.
    with pytest.raises(ConfigError) as info:
        from_dict(HasTrapped, {"x": "unknown"})
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert "resolving enum Trapped raised RuntimeError: members lookup boom" in issue.message


class NumberOnly(Enum):
    ONE = 1


@dataclass
class TwoAnnotatedVariants:
    x: NumberOnly | Annotated[int, "a"] | Annotated[int, "b"]


@dataclass
class TwoAnnotatedVariantsOptional:
    x: NumberOnly | Annotated[int, "a"] | Annotated[int, "b"] | None = None


@pytest.mark.parametrize(
    "config_cls",
    [TwoAnnotatedVariants, TwoAnnotatedVariantsOptional],
    ids=["required", "optional"],
)
def test_several_annotated_variants_of_one_numeric_class_count_once(config_cls: type[Any]):
    # Two members that both strip to int name one numeric class between them, so
    # the union names one of the family and the declared order answers.
    assert from_dict(config_cls, {"x": 1}).x is NumberOnly.ONE


class NameTrap(EnumType):
    """A metaclass that keeps the standard lookup and raises on the class name."""

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("name lookup boom")
        return super().__getattribute__(name)


class TrappedName(Enum, metaclass=NameTrap):
    A = "a"


@dataclass
class HasTrappedName:
    x: TrappedName


def test_a_class_that_declines_to_be_named_still_reports_its_own_problem():
    # Every message naming a class the author owns reads the name through one
    # guarded helper, so a metaclass raising there costs the name rather than the
    # report: the ordinary mismatch still arrives at this value's path.
    with pytest.raises(ConfigError) as info:
        from_dict(HasTrappedName, {"x": "unknown"})
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert issue.message == "expected one of 'a' for enum a class that could not be named, got 'unknown'"
