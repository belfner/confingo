"""Literal arguments, mapping keys, and file-write edges.

Enum-backed Literal arguments, non-finite mapping keys under open data, the
legacy empty-tuple form, umask on a new file, and deterministic set order.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from enum import (
    IntEnum,
    StrEnum,
)
from typing import (  # noqa: UP035  (Tuple used to test the legacy alias)
    TYPE_CHECKING,
    Any,
    Literal,
    Tuple,
)

import pytest

from confingo import (
    ConfigError,
    from_dict,
    load_yaml,
    to_dict,
    to_file,
)


if TYPE_CHECKING:
    from pathlib import Path


# --- enum-backed Literal arguments are rejected as non-primitive ----------


class StrCode(StrEnum):
    A = "a"


class NumCode(IntEnum):
    ONE = 1


@dataclass
class StrLiteral:
    x: Literal[StrCode.A] = StrCode.A


@dataclass
class IntLiteral:
    x: Literal[NumCode.ONE] = NumCode.ONE


@dataclass
class PlainLiteral:
    x: Literal["a", "b"] = "a"


def test_str_enum_literal_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(StrLiteral, {})
    assert any("primitive" in i.message for i in info.value.issues)


def test_int_enum_literal_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(IntLiteral, {})
    assert any("primitive" in i.message for i in info.value.issues)


def test_plain_literal_still_accepted():
    assert from_dict(PlainLiteral, {"x": "b"}).x == "b"


# --- non-finite mapping keys under Any are rejected -----------------------


@dataclass
class AnyHolder:
    x: Any = None


def test_nan_mapping_key_under_any_rejected():
    with pytest.raises(ConfigError):
        from_dict(AnyHolder, {"x": {float("nan"): "v"}})


def test_yaml_nan_key_under_any_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("x:\n  .nan: v\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(AnyHolder, path)


# --- the legacy typing.Tuple[()] empty-tuple form is normalized -----------


@dataclass
class LegacyEmptyTuple:
    x: Tuple[()] = ()  # noqa: UP006  (testing the legacy alias spelling)


def test_legacy_empty_tuple_accepts_empty():
    assert from_dict(LegacyEmptyTuple, {"x": []}).x == ()


def test_legacy_empty_tuple_rejects_nonempty():
    with pytest.raises(ConfigError) as info:
        from_dict(LegacyEmptyTuple, {"x": [1]})
    assert any("expected 0 items" in i.message for i in info.value.issues)


# --- a new file honors the umask (no process-wide umask toggle) -----------


@dataclass
class Payload:
    x: int = 0


def test_atomic_write_new_file_honors_umask(tmp_path: Path):
    old = os.umask(0o022)
    try:
        to_file(Payload(x=1), tmp_path / "new.json")
        assert stat.S_IMODE((tmp_path / "new.json").stat().st_mode) == 0o644
    finally:
        os.umask(old)


# --- set serialization order is stable regardless of element type ---------


@dataclass
class MixedSet:
    s: frozenset[int | str]


def test_mixed_set_serializes_deterministically():
    a = to_dict(MixedSet(frozenset({3, 1, "b", "a"})))
    b = to_dict(MixedSet(frozenset({"a", 3, "b", 1})))
    assert a == b
