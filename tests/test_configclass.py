"""Tests for the ``@configclass`` decorator and canonical-equality injection."""

from __future__ import annotations

import dataclasses
from dataclasses import (
    dataclass,
    field,
)

import pytest

from confingo import (
    ConfigRoot,
    configclass,
    from_dict,
    to_dict,
)
from confingo._configclass import _canonical_eq


# --- schema fixtures (module level so annotations resolve) --------------------


@configclass
class Section:
    name: str = "adam"
    lr: float = 3e-4


@configclass
class Tree(ConfigRoot):
    section: Section
    seed: int = 0


@configclass(frozen=True)
class Frozen:
    x: int = 1


@configclass(slots=True)
class Slotted:
    x: int = 1


@configclass(kw_only=True)
class KwOnly:
    x: int = 1


@configclass
class CustomEq:
    x: int = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomEq)

    __hash__ = object.__hash__


# --- decoration forms and kwargs forwarding -----------------------------------


def test_bare_form_builds_a_dataclass():
    assert dataclasses.is_dataclass(Section)
    assert Section("sgd", 0.1).name == "sgd"


def test_frozen_is_forwarded():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Frozen().x = 2  # pyrefly: ignore[read-only]


def test_slots_is_forwarded():
    assert "__slots__" in Slotted.__dict__
    assert Slotted() == Slotted()


def test_kw_only_is_forwarded():
    with pytest.raises(TypeError):
        KwOnly(1)  # type: ignore[misc]
    assert KwOnly(x=2).x == 2


def test_explicit_eq_raises_type_error():
    with pytest.raises(TypeError, match="eq"):

        @configclass(eq=True)  # type: ignore[call-overload]
        class Bad:
            x: int = 0


def test_order_raises_type_error():
    with pytest.raises(TypeError, match="order"):

        @configclass(order=True)  # type: ignore[call-overload]
        class Ordered:
            x: int = 0


def test_unsafe_hash_raises_type_error():
    with pytest.raises(TypeError, match="unsafe_hash"):

        @configclass(unsafe_hash=True)  # type: ignore[call-overload]
        class Hashed:
            x: int = 0


def test_marker_sits_in_each_classes_own_dict():
    assert Section.__dict__["__confingo_configclass__"] is True
    assert Tree.__dict__["__confingo_configclass__"] is True

    class Child(Section):
        pass

    assert "__confingo_configclass__" not in Child.__dict__


# --- canonical equality -------------------------------------------------------


def test_canonical_equality_across_a_decorated_tree():
    left = Tree(section=Section(), seed=1)
    right = Tree(section=Section(), seed=1)
    assert left == right
    assert left.section == right.section
    assert left != Tree(section=Section(lr=1e-3), seed=1)


def test_round_trip_equality_reads_literally():
    config = Tree(section=Section(name="sgd"), seed=7)
    assert from_dict(Tree, to_dict(config)) == config


def test_foreign_types_get_not_implemented():
    section = Section()
    assert section.__eq__(object()) is NotImplemented
    assert section.__eq__(Frozen()) is NotImplemented
    assert (section == 5) is False
    assert (section != 5) is True


def test_user_defined_eq_is_respected():
    assert Section.__dict__["__eq__"] is Tree.__dict__["__eq__"]
    assert CustomEq.__dict__["__eq__"] is not Section.__dict__["__eq__"]
    assert CustomEq(x=1) == CustomEq(x=2)


def test_hash_stays_object_identity():
    assert Section.__hash__ is object.__hash__
    pair = {Section(), Section()}
    assert len(pair) == 2


def test_frozen_keeps_canonical_eq_and_identity_hash():
    assert Frozen() == Frozen()
    assert Frozen.__hash__ is object.__hash__


# --- canonical-equality injection on plain dataclasses ------------------------


@dataclass
class PlainInjected:
    x: int = 0


@dataclass
class PlainSection:
    y: int = 0


@configclass
class MixedRoot(ConfigRoot):
    section: PlainSection = field(default_factory=PlainSection)
    x: int = 0


@dataclass
class PlainTwin:
    name: str = "adam"
    lr: float = 3e-4


@dataclass
class PlainUserEq:
    x: int = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlainUserEq)

    __hash__ = object.__hash__


@dataclass(eq=False)
class PlainEqFalse:
    x: int = 0


@dataclass(frozen=True)
class PlainFrozen:
    x: int = 0


@dataclass
class PlainTupleField:
    items: tuple[int, ...] = ()


def test_plain_dataclass_gains_canonical_eq_at_first_schema_processing():
    from_dict(PlainInjected, {})
    assert PlainInjected.__dict__["__eq__"] is _canonical_eq
    assert PlainInjected(x=1) == PlainInjected(x=1)
    assert PlainInjected(x=1) != PlainInjected(x=2)


def test_injection_restores_identity_hash():
    from_dict(PlainInjected, {})
    assert PlainInjected.__hash__ is object.__hash__
    pair = {PlainInjected(), PlainInjected()}
    assert len(pair) == 2


def test_plain_section_under_decorated_root_is_injected():
    built = from_dict(MixedRoot, {"section": {"y": 2}})
    assert PlainSection.__dict__["__eq__"] is _canonical_eq
    assert built.section == PlainSection(y=2)


def test_plain_root_compares_across_loads():
    assert from_dict(PlainTwin, {"name": "sgd"}) == from_dict(PlainTwin, {"name": "sgd"})
    assert from_dict(PlainTwin, {"name": "sgd"}) != from_dict(PlainTwin, {"name": "adam"})


def test_class_body_eq_on_a_plain_dataclass_is_kept():
    user_eq = PlainUserEq.__dict__["__eq__"]
    from_dict(PlainUserEq, {})
    assert PlainUserEq.__dict__["__eq__"] is user_eq
    assert PlainUserEq(x=1) == PlainUserEq(x=2)


def test_eq_false_dataclass_gains_canonical_eq():
    assert "__eq__" not in PlainEqFalse.__dict__
    from_dict(PlainEqFalse, {})
    assert PlainEqFalse.__dict__["__eq__"] is _canonical_eq
    assert PlainEqFalse(x=1) == PlainEqFalse(x=1)


def test_plain_frozen_gains_canonical_eq_and_keeps_generated_hash():
    generated_hash = PlainFrozen.__dict__["__hash__"]
    from_dict(PlainFrozen, {})
    assert PlainFrozen.__dict__["__eq__"] is _canonical_eq
    assert PlainFrozen.__dict__["__hash__"] is generated_hash
    assert PlainFrozen(x=1) == PlainFrozen(x=1)


def test_canonical_eq_compares_serialized_container_forms():
    from_dict(PlainTupleField, {"items": [1, 2]})
    assert PlainTupleField(items=(1, 2)) == PlainTupleField(items=[1, 2])  # type: ignore[arg-type]


def test_plain_dataclass_loads_and_dumps_like_a_decorated_one():
    plain = from_dict(PlainTwin, {"name": "sgd"})
    decorated = from_dict(Section, {"name": "sgd"})
    assert to_dict(plain) == to_dict(decorated)
