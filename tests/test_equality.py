"""Tests for canonical equality: ConfigNode installation, injection, and ``config_equal``."""

from __future__ import annotations

import dataclasses
from dataclasses import (
    dataclass,
    field,
)

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    config_equal,
    from_dict,
    to_dict,
)
from confingo._equality import (
    _canonical_eq,
    _unhashable_config,
)


# --- schema fixtures (module level so annotations resolve) --------------------


@dataclass
class Section:
    name: str = "adam"
    lr: float = 3e-4


@dataclass
class Tree(ConfigNode):
    section: Section = field(default_factory=Section)
    seed: int = 0


@dataclass(frozen=True)
class FrozenRoot(ConfigNode):
    x: int = 1


@dataclass(slots=True)
class SlottedRoot(ConfigNode):
    x: int = 1


# --- ConfigNode installs canonical equality at class creation -----------------


def test_root_subclass_carries_canonical_eq_from_class_creation():
    assert Tree.__dict__["__eq__"] is _canonical_eq


def test_root_equality_recurses_before_any_engine_call():
    left = Tree(section=Section(), seed=1)
    right = Tree(section=Section(), seed=1)
    assert left == right
    assert left != Tree(section=Section(lr=1e-3), seed=1)


def test_frozen_root_keeps_canonical_eq_without_a_generated_hash():
    assert FrozenRoot.__dict__["__eq__"] is _canonical_eq
    assert FrozenRoot() == FrozenRoot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        FrozenRoot().x = 2  # pyrefly: ignore[read-only]


def test_slots_root_keeps_canonical_eq_through_class_recreation():
    assert "__slots__" in SlottedRoot.__dict__
    assert SlottedRoot.__dict__["__eq__"] is _canonical_eq
    assert SlottedRoot() == SlottedRoot()


def test_root_body_eq_is_rejected_at_class_creation():
    with pytest.raises(ConfigError, match="custom __eq__"):

        @dataclass
        class CustomEqRoot(ConfigNode):
            x: int = 0

            def __eq__(self, other: object) -> bool:  # pyrefly: ignore[missing-override-decorator]
                return isinstance(other, CustomEqRoot)

            __hash__ = object.__hash__


def test_equal_roots_cannot_enter_a_set():
    with pytest.raises(TypeError, match="unhashable type"):
        {Tree(), Tree()}


def test_config_objects_are_rejected_as_mapping_keys():
    with pytest.raises(TypeError, match="unhashable type"):
        {Tree(): "value"}


def test_foreign_types_get_not_implemented():
    tree = Tree()
    assert tree.__eq__(object()) is NotImplemented  # pyrefly: ignore[bad-argument-type]
    assert (tree == 5) is False
    assert (tree != 5) is True


def test_round_trip_equality_reads_literally():
    config = Tree(section=Section(name="sgd"), seed=7)
    assert from_dict(Tree, to_dict(config)) == config


# --- the two-stage unhashable contract on nodes -------------------------------
#
# Each class below is touched by exactly one test, since the pre-touch stage is
# observable only until the first engine call reaches the class.


@dataclass
class MutableUntouched(ConfigNode):
    x: int = 1


@dataclass(frozen=True)
class FrozenUntouched(ConfigNode):
    x: int = 1


@dataclass(slots=True)
class SlottedUntouched(ConfigNode):
    x: int = 1


@pytest.mark.parametrize("node_cls", [MutableUntouched, FrozenUntouched, SlottedUntouched])
def test_node_sentinel_survives_decoration_then_becomes_none_at_first_touch(node_cls: type[ConfigNode]):
    assert node_cls.__dict__["__hash__"] is _unhashable_config
    with pytest.raises(TypeError, match=r"unhashable type: '\w+'; use config_hash\(config\) for value identity"):
        hash(node_cls())
    node_cls.from_dict({})
    assert node_cls.__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(node_cls())


# --- canonical-equality injection on plain dataclasses ------------------------


@dataclass
class PlainInjected:
    x: int = 0


@dataclass
class PlainSection:
    y: int = 0


@dataclass
class MixedRoot(ConfigNode):
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


def test_injection_disables_hashing():
    from_dict(PlainInjected, {})
    assert PlainInjected.__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        {PlainInjected(), PlainInjected()}


def test_plain_section_under_a_root_is_injected():
    built = from_dict(MixedRoot, {"section": {"y": 2}})
    assert PlainSection.__dict__["__eq__"] is _canonical_eq
    assert built.section == PlainSection(y=2)


def test_plain_root_compares_across_loads():
    assert from_dict(PlainTwin, {"name": "sgd"}) == from_dict(PlainTwin, {"name": "sgd"})
    assert from_dict(PlainTwin, {"name": "sgd"}) != from_dict(PlainTwin, {"name": "adam"})


def test_body_eq_on_a_plain_dataclass_is_rejected():
    with pytest.raises(ConfigError, match="PlainUserEq defines a custom __eq__"):
        from_dict(PlainUserEq, {})


def test_eq_false_dataclass_is_rejected():
    with pytest.raises(ConfigError, match=r"eq=False"):
        from_dict(PlainEqFalse, {})


def test_plain_frozen_gains_canonical_eq_and_loses_its_generated_hash():
    assert PlainFrozen.__dict__["__hash__"] is not None
    from_dict(PlainFrozen, {})
    assert PlainFrozen.__dict__["__eq__"] is _canonical_eq
    assert PlainFrozen.__dict__["__hash__"] is None
    assert PlainFrozen(x=1) == PlainFrozen(x=1)
    with pytest.raises(TypeError, match="unhashable type"):
        hash(PlainFrozen(x=1))


def test_canonical_eq_compares_serialized_container_forms():
    from_dict(PlainTupleField, {"items": [1, 2]})
    assert PlainTupleField(items=(1, 2)) == PlainTupleField(items=[1, 2])  # type: ignore[arg-type]


def test_plain_dataclass_loads_and_dumps_like_a_root_section():
    plain = from_dict(PlainTwin, {"name": "sgd"})
    root = from_dict(Tree, {"section": {"name": "sgd"}})
    assert to_dict(plain) == to_dict(root.section)


# --- config_equal -------------------------------------------------------------


@dataclass
class NeverProcessed:
    items: tuple[int, ...] = ()


def test_config_equal_works_ahead_of_any_engine_call():
    generated_eq = NeverProcessed.__dict__["__eq__"]
    assert config_equal(NeverProcessed(items=(1, 2)), NeverProcessed(items=[1, 2]))  # type: ignore[arg-type]
    assert config_equal(NeverProcessed(items=(1, 2)), NeverProcessed(items=(1, 3))) is False
    assert NeverProcessed.__dict__["__eq__"] is generated_eq


def test_config_equal_applies_the_same_class_rule():
    assert config_equal(Tree(seed=1), Tree(seed=1))
    assert config_equal(Tree(seed=1), Tree(seed=2)) is False
    assert config_equal(Tree(), PlainTwin()) is False
    assert config_equal(Tree(), 5) is False


def test_config_equal_rejects_non_dataclass_input():
    with pytest.raises(TypeError, match="dataclass instance"):
        config_equal(5, Tree())
    with pytest.raises(TypeError, match="dataclass instance"):
        config_equal(Tree, Tree)
