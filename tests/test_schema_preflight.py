"""Schema-level enforcement and value edges, settled ahead of any config data.

Unsupported annotations, dict[Any, X], object-valued enums, and non-finite
floats under open data are each reported for a field the input omits.
"""

from __future__ import annotations

import stat
from collections.abc import Iterable  # noqa: TC003  (needed at runtime by get_type_hints)
from dataclasses import (
    dataclass,
    field,
)
from datetime import date
from enum import (
    Enum,
    EnumType,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NewType,
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
    to_file,
    validate_schema,
)


if TYPE_CHECKING:
    from pathlib import Path

UserId = NewType("UserId", int)


# --- unsupported annotations rejected even when the field is omitted ------


@dataclass
class OmittedIterable:
    xs: Iterable[int] = ()


@dataclass
class OmittedNewType:
    uid: UserId = UserId(0)


def test_unsupported_annotation_rejected_when_omitted():
    with pytest.raises(ConfigError) as info:
        from_dict(OmittedIterable, {})
    assert any("unsupported field type" in i.message for i in info.value.issues)


def test_newtype_rejected_when_omitted():
    with pytest.raises(ConfigError) as info:
        from_dict(OmittedNewType, {})
    assert any("unsupported field type" in i.message for i in info.value.issues)


# --- nested init=False fields are runtime state, excluded from export -----


@dataclass
class WithDerived:
    computed: int = field(init=False, default=1)


@dataclass
class HoldsDerived:
    child: WithDerived = field(default_factory=WithDerived)


def test_nested_init_false_builds_and_is_excluded():
    built = from_dict(HoldsDerived, {})
    assert built.child.computed == 1
    assert to_dict(built) == {"child": {}}


# --- object-valued enums rejected; primitive enums accepted --------------


class ObjectEnum(Enum):
    DAY = date(2020, 1, 1)


class StrEnum(Enum):
    A = "a"
    B = "b"


@dataclass
class HoldsObjectEnum:
    v: ObjectEnum = ObjectEnum.DAY


@dataclass
class HoldsStrEnum:
    v: StrEnum = StrEnum.A


def test_object_valued_enum_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(HoldsObjectEnum, {})
    assert any("primitive" in i.message for i in info.value.issues)


def test_primitive_enum_round_trips():
    cfg = from_dict(HoldsStrEnum, {"v": "b"})
    assert cfg.v is StrEnum.B
    assert from_dict(HoldsStrEnum, to_dict(cfg)) == cfg


# --- dict[Any, X] rejected as a schema ----------------------------------


@dataclass
class DictAnyKey:
    m: dict[Any, int] = field(default_factory=dict)


def test_dict_any_key_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(DictAnyKey, {"m": {}})
    assert any("only str keys" in i.message for i in info.value.issues)


# --- non-finite floats rejected under Any, including nested --------------


@dataclass
class AnyHolder:
    x: ConfigValue = None


def test_nan_under_any_rejected():
    with pytest.raises(ConfigError):
        from_dict(AnyHolder, {"x": float("nan")})


def test_nested_infinity_under_any_rejected():
    with pytest.raises(ConfigError):
        from_dict(AnyHolder, {"x": {"nested": [float("inf")]}})


def test_plain_data_under_any_still_works():
    cfg = from_dict(AnyHolder, {"x": {"a": [1, 2.5, "three"]}})
    assert cfg.x == {"a": [1, 2.5, "three"]}


# --- tuple[()] enforces the empty tuple ---------------------------------


@dataclass
class EmptyTuple:
    x: tuple[()] = ()


def test_empty_tuple_accepts_empty():
    assert from_dict(EmptyTuple, {"x": []}).x == ()


def test_empty_tuple_rejects_nonempty():
    with pytest.raises(ConfigError) as info:
        from_dict(EmptyTuple, {"x": [1, 2]})
    assert any("expected 0 items" in i.message for i in info.value.issues)


# --- single-type optional validates in one pass --------------------------

_post_init_calls = {"n": 0}


@dataclass
class Validated:
    x: int = 0

    def __post_init__(self):
        _post_init_calls["n"] += 1
        if self.x < 0:
            raise ValueError("x must be non-negative")


@dataclass
class OptionalValidated:
    child: Validated | None = None


def test_optional_runs_post_init_once_on_failure():
    _post_init_calls["n"] = 0
    with pytest.raises(ConfigError):
        from_dict(OptionalValidated, {"child": {"x": -1}})
    assert _post_init_calls["n"] == 1


# --- atomic write preserves the destination's file mode ------------------


@dataclass
class Payload:
    x: int = 0


def test_atomic_write_preserves_existing_mode(tmp_path: Path):
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o644)
    to_file(Payload(x=1), target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


class PreflightNameTrap(EnumType):
    """A metaclass that keeps the standard lookup and raises on the class name."""

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("name lookup boom")
        return super().__getattribute__(name)


class UnnameableEnum(Enum, metaclass=PreflightNameTrap):
    A = ("not", "primitive")


@dataclass
class HasUnnameableEnum:
    x: UnnameableEnum


@pytest.mark.parametrize(
    "operation",
    [validate_schema, lambda config_cls: from_dict(config_cls, {"x": 1})],
    ids=["validate_schema", "from_dict"],
)
def test_a_schema_issue_naming_an_unnameable_class_still_reports_its_own_problem(operation: Any):
    # Preflight has its own problem to report about this enum's member value.
    # Reading the class name goes through the metaclass, so it reads through the
    # one guarded helper and the member-value report still arrives.
    with pytest.raises(ConfigError) as info:
        operation(HasUnnameableEnum)
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert issue.message.startswith("enum a class that could not be named must carry primitive values")
    assert "('not', 'primitive')" in issue.message
