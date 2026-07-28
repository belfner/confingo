"""Tests for dataclass ``field()`` options: init=False, compare, and hash.

``init`` is the master switch: an ``init=False`` field is excluded from loading,
export, equality, and the fingerprint, and is populated in ``__post_init__``
(checked for completeness after construction). On an ``init=True`` field
``compare`` scopes equality (and the fingerprint) and ``hash`` scopes the
fingerprint, with ``hash=True, compare=False`` the sole rejected contradiction.
"""

from __future__ import annotations

import logging
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    ConfigValue,
)
from confingo.functional import (
    config_equal,
    config_hash,
    from_dict,
    to_dict,
)


# --- init=False: runtime state populated in __post_init__ --------------------


@dataclass
class WithCache:
    a: int
    cache: object = field(init=False)

    def __post_init__(self) -> None:
        self.cache = object()


def test_init_false_populated_and_excluded():
    built = from_dict(WithCache, {"a": 5})
    assert built.a == 5
    assert isinstance(built.cache, object)
    assert to_dict(built) == {"a": 5}


def test_init_false_ignored_by_equality_and_hash():
    left = from_dict(WithCache, {"a": 5})
    right = from_dict(WithCache, {"a": 5})
    # Distinct cache objects, yet equal and same fingerprint.
    assert left.cache is not right.cache
    assert config_equal(left, right)
    assert config_hash(left) == config_hash(right)


def test_init_false_round_trips():
    built = from_dict(WithCache, {"a": 5})
    assert config_equal(from_dict(WithCache, to_dict(built)), built)


def test_init_false_key_not_configurable():
    with pytest.raises(ConfigError) as info:
        from_dict(WithCache, {"a": 5, "cache": 1})
    messages = [str(i) for i in info.value.issues]
    assert "cache: field is not configurable (init=False)" in messages


@dataclass
class MultiIssue:
    a: int  # required; supplied a bad value -> coercion failure
    b: int  # required; omitted -> missing value
    cache: object = field(init=False)  # runtime; supplying its key -> not configurable

    def __post_init__(self) -> None:
        self.cache = object()


def test_non_configurable_key_collected_with_other_issues():
    # The pre-construction key walk records the not-configurable and unknown-key
    # issues; the field walk records the coercion and missing-value issues. All
    # surface in one collect-all report.
    with pytest.raises(ConfigError) as info:
        from_dict(MultiIssue, {"a": "bad", "cache": 1, "typo": 5})
    paths = {i.path for i in info.value.issues}
    assert {"a", "b", "cache", "typo"} <= paths
    assert "cache: field is not configurable (init=False)" in [str(i) for i in info.value.issues]


if TYPE_CHECKING:

    class _RuntimeOnlyType: ...


@dataclass
class UnresolvableRuntime:
    a: int
    handle: _RuntimeOnlyType = field(init=False)  # resolvable only under TYPE_CHECKING

    def __post_init__(self) -> None:
        self.handle = None  # type: ignore[assignment]


def test_init_false_annotation_must_still_resolve():
    # init=False exempts a field from the supported-type boundary, but its
    # annotation must still resolve at runtime; a TYPE_CHECKING-only name does not.
    with pytest.raises(ConfigError) as info:
        from_dict(UnresolvableRuntime, {"a": 1})
    assert any("resolve" in str(i).lower() for i in info.value.issues)


@dataclass
class RuntimeObjects:
    a: int
    logger: logging.Logger = field(init=False)
    handle: object = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger("confingo.test")
        self.handle = object()


def test_init_false_annotation_boundary_exempt():
    # logging.Logger and object are outside the supported leaf set, allowed only
    # because the fields are init=False runtime state.
    built = from_dict(RuntimeObjects, {"a": 1})
    assert built.logger.name == "confingo.test"
    assert to_dict(built) == {"a": 1}


# --- init=False flags are inert (all spellings behave identically) -----------


@dataclass
class InitFalseBare:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        self.b = self.a * 2


@dataclass
class InitFalseCompareTrue:
    a: int
    b: int = field(init=False, compare=True)

    def __post_init__(self) -> None:
        self.b = self.a * 2


@dataclass
class InitFalseCompareFalse:
    a: int
    b: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        self.b = self.a * 2


@dataclass
class InitFalseHashTrue:
    a: int
    b: int = field(init=False, hash=True)

    def __post_init__(self) -> None:
        self.b = self.a * 2


@pytest.mark.parametrize("cls", [InitFalseBare, InitFalseCompareTrue, InitFalseCompareFalse, InitFalseHashTrue])
def test_init_false_flags_are_inert(cls: type[Any]):
    built = from_dict(cls, {"a": 3})
    other = from_dict(cls, {"a": 3})
    assert built.b == 6
    assert to_dict(built) == {"a": 3}  # never exported
    # Force the runtime field to differ. If compare=True / hash=True had leaked
    # it into equality or the fingerprint, these would fail; init=False keeps
    # the flags inert.
    built.b = 100
    other.b = 200
    assert config_equal(built, other)
    assert config_hash(built) == config_hash(other)


# --- init=False defaults and default_factory ---------------------------------


@dataclass
class InitFalseDefault:
    a: int
    b: int = field(init=False, default=7)
    c: list[int] = field(init=False, default_factory=list)


def test_init_false_default_and_factory_satisfy_completeness():
    built = from_dict(InitFalseDefault, {"a": 1})
    assert built.b == 7
    assert built.c == []
    assert to_dict(built) == {"a": 1}


_factory_calls: list[int] = []


def _counting_factory() -> list[int]:
    _factory_calls.append(1)
    return []


@dataclass
class InitFalseCountedFactory:
    a: int
    c: list[int] = field(init=False, default_factory=_counting_factory)


def test_init_false_default_factory_invoked_once():
    _factory_calls.clear()
    built = from_dict(InitFalseCountedFactory, {"a": 1})
    assert built.c == []
    # The dataclass constructor applies the factory exactly once; the engine adds
    # no second invocation.
    assert len(_factory_calls) == 1


# --- completeness check ------------------------------------------------------


@dataclass
class NeverPopulates:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        # Never assigns self.b.
        return


def test_completeness_unset_field_reported():
    with pytest.raises(ConfigError) as info:
        from_dict(NeverPopulates, {"a": 5})
    assert [str(i) for i in info.value.issues] == ["b: init=False field was not set during __post_init__"]


@dataclass
class ValidateReadsUnset:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        return

    def __validate__(self) -> list[str]:
        # Would raise AttributeError if invoked on an incomplete instance.
        return [] if self.b > 0 else ["b must be positive"]


def test_completeness_skips_validate_no_attributeerror_leak():
    with pytest.raises(ConfigError) as info:
        from_dict(ValidateReadsUnset, {"a": 5})
    # The completeness issue is reported; __validate__ was not invoked.
    assert [str(i) for i in info.value.issues] == ["b: init=False field was not set during __post_init__"]


@dataclass
class TwoUnset:
    a: int
    b: int = field(init=False)
    c: int = field(init=False)

    def __post_init__(self) -> None:
        return


def test_completeness_reports_multiple_unset_together():
    with pytest.raises(ConfigError) as info:
        from_dict(TwoUnset, {"a": 5})
    messages = {str(i) for i in info.value.issues}
    assert messages == {
        "b: init=False field was not set during __post_init__",
        "c: init=False field was not set during __post_init__",
    }


@dataclass
class UnsetPlusUnknownKey:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        return


def test_completeness_joins_unknown_key():
    # The unknown-key issue is recorded before construction; the completeness
    # issue after. Both surface in one report.
    with pytest.raises(ConfigError) as info:
        from_dict(UnsetPlusUnknownKey, {"a": 5, "typo": 1})
    messages = {str(i) for i in info.value.issues}
    assert "b: init=False field was not set during __post_init__" in messages
    assert any("unknown key" in m for m in messages)


@dataclass
class UnsetPlusCoercionFailure:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        return


def test_completeness_not_reported_with_loadable_coercion_failure():
    # A loadable coercion failure short-circuits before construction, so
    # __post_init__ and the completeness check never run: only the coercion
    # issue appears, not the completeness one.
    with pytest.raises(ConfigError) as info:
        from_dict(UnsetPlusCoercionFailure, {"a": "not-an-int"})
    messages = [str(i) for i in info.value.issues]
    assert not any("was not set during __post_init__" in m for m in messages)
    assert len(messages) == 1


@dataclass
class InnerRuntime:
    x: int
    handle: int = field(init=False)

    def __post_init__(self) -> None:
        return


@dataclass
class OuterRuntime:
    inner: InnerRuntime


def test_completeness_reported_at_nested_path():
    with pytest.raises(ConfigError) as info:
        from_dict(OuterRuntime, {"inner": {"x": 1}})
    assert [str(i) for i in info.value.issues] == ["inner.handle: init=False field was not set during __post_init__"]


@dataclass(frozen=True)
class FrozenRuntime:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "b", self.a + 1)


def test_completeness_frozen_class():
    built = from_dict(FrozenRuntime, {"a": 4})
    assert built.b == 5
    assert to_dict(built) == {"a": 4}


@dataclass(slots=True)
class SlotsRuntime:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        return


def test_completeness_slots_class_detects_unset():
    with pytest.raises(ConfigError) as info:
        from_dict(SlotsRuntime, {"a": 4})
    assert [str(i) for i in info.value.issues] == ["b: init=False field was not set during __post_init__"]


@dataclass
class GetattrFallback:
    a: int
    b: int = field(init=False)

    def __post_init__(self) -> None:
        return

    def __getattr__(self, name: str) -> int:
        # A synthetic fallback that hasattr() would be fooled by.
        return 999


def test_completeness_not_fooled_by_getattr():
    with pytest.raises(ConfigError) as info:
        from_dict(GetattrFallback, {"a": 4})
    assert [str(i) for i in info.value.issues] == ["b: init=False field was not set during __post_init__"]


# --- compare=False (init=True) -----------------------------------------------


@dataclass
class HasCompareFalse:
    a: int
    note: str = field(default="x", compare=False)


def test_compare_false_excluded_from_equality_and_hash_but_exported():
    left = from_dict(HasCompareFalse, {"a": 1, "note": "hello"})
    right = from_dict(HasCompareFalse, {"a": 1, "note": "world"})
    assert to_dict(left) == {"a": 1, "note": "hello"}  # still exported
    assert left == right  # operator __eq__
    assert config_equal(left, right)
    assert config_hash(left) == config_hash(right)


def test_compare_false_round_trips():
    built = from_dict(HasCompareFalse, {"a": 1, "note": "hello"})
    assert config_equal(from_dict(HasCompareFalse, to_dict(built)), built)


@dataclass
class Section:
    a: int
    note: str = field(default="x", compare=False)


@dataclass
class HoldsSection:
    section: Section


def test_compare_false_recurses_into_sections():
    left = from_dict(HoldsSection, {"section": {"a": 1, "note": "p"}})
    right = from_dict(HoldsSection, {"section": {"a": 1, "note": "q"}})
    assert config_equal(left, right)
    assert config_hash(left) == config_hash(right)


@dataclass
class HoldsSectionTuple:
    sections: tuple[Section, ...] = ()


def test_compare_false_recurses_through_a_section_collection():
    left = from_dict(HoldsSectionTuple, {"sections": [{"a": 1, "note": "p"}]})
    right = from_dict(HoldsSectionTuple, {"sections": [{"a": 1, "note": "q"}]})
    assert config_equal(left, right)
    assert config_hash(left) == config_hash(right)


@dataclass
class HoldsAnyMapping:
    mapping: ConfigValue = None


def test_compare_false_drops_through_the_serialized_comparison_fallback():
    # A dataclass reached inside a mapping with non-str keys compares by its
    # COMPARE-projection plain form rather than by structural recursion, and that
    # projection drops the compare=False field the EXPORT projection keeps.
    # An int key and a section value both sit outside ConfigValue; the fallback
    # under test is reached only by a value the annotation declines.
    left = HoldsAnyMapping(mapping={1: Section(a=1, note="p")})  # pyrefly: ignore[bad-argument-type]
    right = HoldsAnyMapping(mapping={1: Section(a=1, note="q")})  # pyrefly: ignore[bad-argument-type]
    assert config_equal(left, right)
    assert config_hash(left) == config_hash(right)
    assert to_dict(left) == {"mapping": {1: {"a": 1, "note": "p"}}}


# --- hash=False (init=True) --------------------------------------------------


@dataclass
class HasHashFalse:
    a: int
    seed: int = field(default=0, hash=False)


def test_hash_false_kept_in_equality_dropped_from_fingerprint():
    left = from_dict(HasHashFalse, {"a": 1, "seed": 10})
    right = from_dict(HasHashFalse, {"a": 1, "seed": 20})
    assert to_dict(left) == {"a": 1, "seed": 10}  # still exported
    assert not config_equal(left, right)  # seed differs -> unequal
    assert config_hash(left) == config_hash(right)  # but same fingerprint


@dataclass
class SeedSection:
    a: int
    seed: int = field(default=0, hash=False)


@dataclass
class HoldsSeedSection:
    section: SeedSection


def test_hash_false_recurses_into_sections():
    left = from_dict(HoldsSeedSection, {"section": {"a": 1, "seed": 10}})
    right = from_dict(HoldsSeedSection, {"section": {"a": 1, "seed": 20}})
    assert not config_equal(left, right)  # nested seed differs -> unequal
    assert config_hash(left) == config_hash(right)  # excluded from digest recursively


# --- equality tracks the fingerprint token-for-token --------------------------


@dataclass
class AnyBox:
    x: ConfigValue


@pytest.mark.parametrize(
    ("a", "b"),
    [(0.0, -0.0), ([0.0], [-0.0]), ({"k": 0.0}, {"k": -0.0}), (1, 1.0), (True, 1), (True, 1.0)],
)
def test_cross_token_values_are_unequal_and_hash_apart(a: Any, b: Any):
    # Values whose canonical JSON differs (0.0 vs -0.0, 1 vs 1.0, true vs 1) must
    # compare unequal so config_equal never disagrees with config_hash.
    assert not config_equal(AnyBox(a), AnyBox(b))
    assert config_hash(AnyBox(a)) != config_hash(AnyBox(b))


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, [1.5, -0.0], {"k": 1.0}])
def test_same_token_values_stay_equal(value: Any):
    assert config_equal(AnyBox(value), AnyBox(value))
    assert config_hash(AnyBox(value)) == config_hash(AnyBox(value))


def test_non_finite_scalar_floats_compare_without_raising():
    # A non-finite float has no plain form (config_hash would reject it), so
    # config_equal falls back to Python float semantics rather than raising.
    assert config_equal(AnyBox(float("inf")), AnyBox(float("inf")))
    assert not config_equal(AnyBox(float("nan")), AnyBox(float("nan")))


# --- valid init=True pairings ------------------------------------------------


@dataclass
class HashTrueCompareTrue:
    a: int
    b: int = field(default=0, hash=True, compare=True)


def test_hash_true_compare_true_included_everywhere():
    left = from_dict(HashTrueCompareTrue, {"a": 1, "b": 2})
    right = from_dict(HashTrueCompareTrue, {"a": 1, "b": 3})
    assert not config_equal(left, right)
    assert config_hash(left) != config_hash(right)


@pytest.mark.parametrize(
    ("compare", "hash_"),
    [(False, None), (False, False), (True, False), (True, None), (True, True)],
)
def test_valid_init_true_pairings_do_not_raise(compare: bool, hash_: bool | None):
    @dataclass
    class Pairing:
        a: int
        b: int = field(default=0, compare=compare, hash=hash_)

    # Building must not raise a preflight conflict for any valid pairing.
    assert from_dict(Pairing, {"a": 1}).a == 1


# --- the one rejected contradiction ------------------------------------------


@dataclass
class Contradiction:
    a: int
    x: int = field(default=0, hash=True, compare=False)


def test_hash_true_compare_false_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(Contradiction, {"a": 1})
    assert [str(i) for i in info.value.issues] == [
        "x: field(hash=True, compare=False) is contradictory: config_hash fields must participate in equality"
    ]


@dataclass
class InitFalseSameFlags:
    a: int
    x: int = field(init=False, hash=True, compare=False)

    def __post_init__(self) -> None:
        self.x = self.a


def test_init_false_makes_the_contradiction_inert():
    # The same flags that are rejected on an init=True field are inert here.
    built = from_dict(InitFalseSameFlags, {"a": 1})
    assert built.x == 1
    assert to_dict(built) == {"a": 1}


# --- projection integrity ----------------------------------------------------


@dataclass
class DeclarationOrder:
    a: int
    b: str = field(default="b", compare=False)
    c: float = field(default=1.0, hash=False)
    d: int = field(init=False, default=0)


def test_to_dict_emits_all_init_true_in_declaration_order():
    built = from_dict(DeclarationOrder, {"a": 1})
    plain = to_dict(built)
    assert list(plain.keys()) == ["a", "b", "c"]  # d (init=False) omitted


# --- ConfigNode / plain parity -----------------------------------------------


@dataclass
class PlainRoot:
    a: int
    note: str = field(default="x", compare=False)


@dataclass
class RootSubclass(ConfigNode):
    a: int
    note: str = field(default="x", compare=False)


@pytest.mark.parametrize("cls", [PlainRoot, RootSubclass])
def test_compare_false_parity_plain_and_configroot(cls: type[Any]):
    left = from_dict(cls, {"a": 1, "note": "p"})
    right = from_dict(cls, {"a": 1, "note": "q"})
    assert config_equal(left, right)
    assert left == right
    assert config_hash(left) == config_hash(right)


@dataclass
class UntouchedPlain:
    a: int
    note: str = field(default="x", compare=False)


@dataclass
class UntouchedRoot(ConfigNode):
    a: int
    note: str = field(default="x", compare=False)


def test_config_equal_before_any_engine_call():
    # config_equal resolves the classification at call time, so it honors
    # compare=False on freshly constructed instances of a plain dataclass and a
    # ConfigNode subclass alike, ahead of any from_dict/to_dict call. A
    # ConfigNode installs canonical __eq__ at class creation, so `==` works too.
    assert config_equal(UntouchedPlain(1, "p"), UntouchedPlain(1, "q"))
    assert config_equal(UntouchedRoot(1, "p"), UntouchedRoot(1, "q"))
    assert UntouchedRoot(1, "p") == UntouchedRoot(1, "q")


# --- Python hashing is disabled, config_hash carries value identity -----------


def test_python_hash_is_disabled():
    # Processing the schema installs canonical equality and makes the class
    # unhashable, so value identity comes from config_hash instead.
    processed = from_dict(PlainRoot, {"a": 1, "note": "p"})
    assert PlainRoot.__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(processed)
    other = from_dict(PlainRoot, {"a": 1, "note": "q"})
    assert config_equal(processed, other)
    assert config_hash(processed) == config_hash(other)
