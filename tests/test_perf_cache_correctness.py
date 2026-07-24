"""Correctness guards for the hint-plan cache and marshal fast paths.

These tests target the ways type caching and the exact-scalar fast paths could
break normal usage: cache-enabled and cache-disabled runs must agree; unhashable
hints must classify without raising; identity keys must never alias distinct
same-named classes; a bounded cache must release dynamically created classes;
IntEnum/StrEnum/bool leaves must keep enum/scalar semantics; and non-finite
floats must still be rejected.
"""

from __future__ import annotations

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import (
    dataclass,
    make_dataclass,
)
from enum import (
    Enum,
    IntEnum,
    StrEnum,
)
from threading import Barrier
from typing import (
    Annotated,
    Any,
)

import pytest

import confingo
from confingo import (
    ConfigError,
    _schema,
    config_hash,
    from_dict,
    to_dict,
)


@dataclass
class Leaf:
    name: str
    value: int


@dataclass
class Branch:
    leaf: Leaf
    tags: list[str]
    weight: float | None


@dataclass
class Tree:
    root: Branch
    leaves: list[Leaf]
    labels: dict[str, int]


_VALID = {
    "root": {"leaf": {"name": "a", "value": 1}, "tags": ["x", "y"], "weight": 0.5},
    "leaves": [{"name": "b", "value": 2}, {"name": "c", "value": 3}],
    "labels": {"k": 4},
}

_INVALID = {
    "root": {"leaf": {"name": "a"}, "tags": ["x"], "weight": "nope"},
    "leaves": [{"name": "b", "value": "bad"}],
    "labels": {"k": 4},
}


def _run_corpus() -> tuple[Any, str, list[str]]:
    """Load valid and invalid inputs, returning marshalled data, hash, and issues.

    Returns:
      tuple[Any, str, list[str]]: The plain dict of the valid build, its config
        hash, and the sorted dotted "path: message" issue lines from the invalid
        build.
    """
    built = from_dict(Tree, _VALID)
    plain = to_dict(built)
    digest = config_hash(built)
    with pytest.raises(ConfigError) as exc:
        from_dict(Tree, _INVALID)
    issues = sorted(f"{i.path}: {i.message}" for i in exc.value.issues)
    return plain, digest, issues


def test_cache_disabled_matches_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache-enabled and cache-disabled runs produce identical output and issues."""
    _schema._classify_hint_by_id.cache_clear()
    monkeypatch.setattr(_schema, "_TYPE_CACHE_DISABLED", False)
    enabled = _run_corpus()

    monkeypatch.setattr(_schema, "_TYPE_CACHE_DISABLED", True)
    disabled = _run_corpus()

    assert enabled == disabled


def test_cache_and_uncached_classification_agree() -> None:
    """The cached classifier matches the uncached one field for field."""
    for hint in (int, str, float | None, list[Leaf], dict[str, int], Leaf, Annotated[int, "m"]):
        cached = _schema._classify_hint(hint)
        uncached = _schema._classify_hint_uncached(hint)
        assert cached.kind is uncached.kind
        assert cached.stripped == uncached.stripped
        assert cached.args == uncached.args


def test_unhashable_annotated_metadata_loads() -> None:
    """A field whose Annotated metadata is unhashable still classifies and loads."""

    @dataclass
    class HasUnhashable:
        x: Annotated[int, []]  # list metadata is unhashable
        y: Annotated[str, {"k": "v"}]  # dict metadata is unhashable

    built = from_dict(HasUnhashable, {"x": 7, "y": "hello"})
    assert built == HasUnhashable(x=7, y="hello")
    assert to_dict(built) == {"x": 7, "y": "hello"}


def test_identity_cache_distinguishes_same_named_classes() -> None:
    """Two distinct dataclasses sharing a name build their own types, never aliased."""
    dup_int = make_dataclass("Dup", [("v", int)])
    dup_str = make_dataclass("Dup", [("v", str)])
    parent_int = make_dataclass("ParentA", [("d", dup_int)])
    parent_str = make_dataclass("ParentB", [("d", dup_str)])

    built_int = from_dict(parent_int, {"d": {"v": 5}})
    built_str = from_dict(parent_str, {"d": {"v": "five"}})

    assert type(built_int.d) is dup_int
    assert type(built_str.d) is dup_str
    assert built_int.d.v == 5
    assert built_str.d.v == "five"


def test_redefined_class_gets_fresh_plan() -> None:
    """Redefining a class name yields a new type whose plan is not the stale one."""
    first = make_dataclass("Redef", [("v", int)])
    assert from_dict(first, {"v": 1}).v == 1

    second = make_dataclass("Redef", [("v", str)])
    assert from_dict(second, {"v": "two"}).v == "two"
    # The int-typed redefinition must still reject a str, proving no stale plan.
    with pytest.raises(ConfigError):
        from_dict(first, {"v": "not-an-int"})


class Color(IntEnum):
    red = 1
    green = 2


class Size(StrEnum):
    small = "s"
    large = "l"


class Plain(Enum):
    a = "a"
    b = "b"


@dataclass
class EnumLeaves:
    color: Color
    size: Size
    plain: Plain
    flag: bool
    count: int
    name: str


def test_intenum_strenum_bool_leaves_roundtrip() -> None:
    """IntEnum/StrEnum keep enum semantics and bool stays bool through the fast path."""
    data = {"color": 1, "size": "l", "plain": "a", "flag": True, "count": 3, "name": "n"}
    built = from_dict(EnumLeaves, data)
    assert built.color is Color.red
    assert built.size is Size.large
    assert built.flag is True

    plain = to_dict(built)
    # Enum members marshal to their primitive .value, not the member object.
    assert plain == {"color": 1, "size": "l", "plain": "a", "flag": True, "count": 3, "name": "n"}
    assert type(plain["flag"]) is bool
    assert from_dict(EnumLeaves, plain) == built


@dataclass
class FloatBox:
    ratio: float
    anything: Any


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_float_rejected_on_marshal(bad: float) -> None:
    """Exact non-finite floats are rejected by to_dict at their dotted path."""
    box = FloatBox(ratio=bad, anything=1)
    with pytest.raises(ConfigError) as exc:
        to_dict(box)
    assert any(i.path == "ratio" for i in exc.value.issues)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_float_rejected_under_any(bad: float) -> None:
    """Non-finite floats nested under an Any field are still rejected."""
    box = FloatBox(ratio=1.0, anything={"deep": bad})
    with pytest.raises(ConfigError):
        to_dict(box)


def test_hint_plan_cache_is_bounded_and_releases_classes() -> None:
    """The hint-plan cache stays within maxsize and lets evicted classes be collected."""
    _schema._classify_hint_by_id.cache_clear()
    maxsize = _schema._classify_hint_by_id.cache_info().maxsize
    assert maxsize is not None

    overflow = maxsize + 200
    first_cls = make_dataclass("Leaf0", [("v", int)])
    first_ref = weakref.ref(first_cls)
    _schema._classify_hint(list[first_cls])
    del first_cls

    for i in range(1, overflow):
        cls = make_dataclass(f"Leaf{i}", [("v", int)])
        _schema._classify_hint(list[cls])
        _schema._classify_hint(cls)

    info = _schema._classify_hint_by_id.cache_info()
    assert info.currsize <= maxsize

    gc.collect()
    # The earliest class was evicted long ago, so nothing pins it anymore.
    assert first_ref() is None


def test_concurrent_first_touch_is_consistent() -> None:
    """Many threads first-touching a fresh schema all build identical results."""
    fresh = make_dataclass("Concurrent", [("v", int), ("w", str)])
    _schema._classify_hint_by_id.cache_clear()
    workers = 24
    barrier = Barrier(workers)

    def build(_: int) -> Any:
        barrier.wait()
        return from_dict(fresh, {"v": 10, "w": "x"})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(build, range(workers)))

    assert all(r == results[0] for r in results)
    assert results[0].v == 10
    assert results[0].w == "x"


def test_base_package_stays_stdlib_only() -> None:
    """Importing and using confingo never imports the optional array backends."""
    # The test session may have numpy/torch loaded from array tests; only assert
    # confingo itself does not require them for a plain schema.
    built = from_dict(Tree, _VALID)
    assert to_dict(built)["labels"] == {"k": 4}
    assert confingo.config_hash(built)
    # A no-backend build path is exercised whenever these modules are absent.
    if "numpy" not in sys.modules and "torch" not in sys.modules:
        assert built.root.leaf.name == "a"
