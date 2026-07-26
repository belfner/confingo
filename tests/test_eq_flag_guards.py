"""Tests for the ownership guard: custom ``__eq__`` / ``__hash__`` and conflicting flags.

confingo owns equality and hashing on config dataclasses. A hand-written ``__eq__``
or ``__hash__``, or a ``@dataclass`` flag confingo cannot honor, is rejected with a
``ConfigError`` -- roots at class creation for a body method, everything else at the
first schema touch. At that touch the class becomes unhashable, whatever ``__hash__``
it arrived with.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

import numpy as np
import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    config_equal,
    config_hash,
    from_dict,
    to_dict,
)
from confingo._equality import _unhashable_config


# --- module-level fixtures (annotations resolve at module scope) --------------


@dataclass
class SectionCustomEq:
    a: int = 0

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = object.__hash__


@dataclass
class RootWithCustomEqSection(ConfigNode):
    sec: SectionCustomEq = field(default_factory=SectionCustomEq)


@dataclass
class SectionCustomHash:
    a: int = 0

    def __hash__(self) -> int:
        return 17


@dataclass
class RootWithCustomHashSection(ConfigNode):
    sec: SectionCustomHash = field(default_factory=SectionCustomHash)


@dataclass(order=True)
class OrderSection:
    a: int = 0


@dataclass
class RootWithOrderSection(ConfigNode):
    sec: OrderSection = field(default_factory=OrderSection)


@dataclass(eq=False)
class NoEqSection:
    a: int = 0


@dataclass
class RootWithNoEqSection(ConfigNode):
    sec: NoEqSection = field(default_factory=NoEqSection)


@dataclass(unsafe_hash=True)
class UnsafeHashSection:
    a: int = 0


@dataclass
class RootWithUnsafeHashSection(ConfigNode):
    sec: UnsafeHashSection = field(default_factory=UnsafeHashSection)


@dataclass(init=False)
class NoInitSection:
    a: int = 5


@dataclass
class RootWithNoInitSection(ConfigNode):
    sec: NoInitSection = field(default_factory=NoInitSection)


@dataclass(order=True, unsafe_hash=True)
class MultiFlagSection:
    a: int = 0


@dataclass
class RootWithMultiFlagSection(ConfigNode):
    sec: MultiFlagSection = field(default_factory=MultiFlagSection)


# inherited conflicts via undecorated subclasses


@dataclass(eq=False)
class EqFalseBase:
    a: int = 0


class InheritsEqFalse(EqFalseBase):
    pass


@dataclass
class RootInheritsEqFalse(ConfigNode):
    sec: InheritsEqFalse = field(default_factory=InheritsEqFalse)  # pyrefly: ignore[bad-assignment]


@dataclass
class CustomEqBase:
    a: int = 0

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = object.__hash__


class InheritsCustomEq(CustomEqBase):
    pass


@dataclass
class RootInheritsCustomEq(ConfigNode):
    sec: InheritsCustomEq = field(default_factory=InheritsCustomEq)  # pyrefly: ignore[bad-assignment]


# a section whose __eq__ is a C-level slot wrapper (no __code__)


@dataclass
class SlotWrapperEqSection:
    a: int = 0
    __eq__ = str.__eq__
    __hash__ = object.__hash__


@dataclass
class RootWithSlotWrapperEq(ConfigNode):
    sec: SlotWrapperEqSection = field(default_factory=SlotWrapperEqSection)


# good schemas


@dataclass(frozen=True)
class FrozenSection:
    a: int = 0
    b: float = 1.5


@dataclass
class GoodRoot(ConfigNode):
    fs: FrozenSection = field(default_factory=FrozenSection)
    plain: int = 3


@dataclass(frozen=True)
class FrozenArraySection:
    arr: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))


@dataclass
class RootWithFrozenArray(ConfigNode):
    fa: FrozenArraySection = field(default_factory=FrozenArraySection)


@dataclass(slots=True)
class SlotsInner:
    lr: float
    steps: int = 10
    derived: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", self.lr * self.steps)


@dataclass(slots=True)
class SlotsRoot(ConfigNode):
    inner: SlotsInner
    name: str = "run"


@dataclass(slots=True)
class SlotsIncompleteRoot(ConfigNode):
    v: int = 0
    missing: int = field(init=False)


@dataclass(slots=True, weakref_slot=True)
class SlotsWeakrefRoot(ConfigNode):
    x: int = 1


# --- custom __eq__ ------------------------------------------------------------


def test_custom_eq_root_rejected_at_class_creation():
    with pytest.raises(ConfigError, match="custom __eq__"):

        @dataclass
        class CustomEqRoot(ConfigNode):
            x: int = 0

            def __eq__(self, other: object) -> bool:  # pyrefly: ignore[missing-override-decorator]
                return True

            __hash__ = object.__hash__


def test_custom_eq_section_rejected_at_first_touch():
    with pytest.raises(ConfigError, match="SectionCustomEq defines a custom __eq__"):
        from_dict(RootWithCustomEqSection, {})


def test_inherited_custom_eq_rejected_via_mro():
    with pytest.raises(ConfigError, match="CustomEqBase defines a custom __eq__"):
        from_dict(RootInheritsCustomEq, {})


def test_slot_wrapper_eq_without_code_treated_as_custom():
    with pytest.raises(ConfigError, match="SlotWrapperEqSection defines a custom __eq__"):
        from_dict(RootWithSlotWrapperEq, {})


# --- custom __hash__ ----------------------------------------------------------


def test_custom_hash_only_root_rejected_at_class_creation():
    with pytest.raises(ConfigError, match="custom __hash__"):

        @dataclass
        class CustomHashRoot(ConfigNode):
            x: int = 0

            def __hash__(self) -> int:
                return 17


def test_custom_hash_only_slots_root_rejected_at_class_creation():
    with pytest.raises(ConfigError, match="custom __hash__"):

        @dataclass(slots=True)
        class CustomHashSlotsRoot(ConfigNode):
            x: int = 0

            def __hash__(self) -> int:
                return 17


def test_custom_hash_section_rejected_at_first_touch():
    with pytest.raises(ConfigError, match="SectionCustomHash defines a custom __hash__"):
        from_dict(RootWithCustomHashSection, {})


def test_custom_hash_message_names_config_hash():
    with pytest.raises(ConfigError, match="use config_hash\\(config\\) for value identity"):
        from_dict(RootWithCustomHashSection, {})


# --- confingo's own hashes are told apart from hand-written ones ---------------


@dataclass
class SentinelBase(ConfigNode):
    x: int = 0


@dataclass
class SentinelDerived(SentinelBase):
    y: int = 0


@dataclass
class TouchedBase(ConfigNode):
    x: int = 0


def test_node_subclass_reads_an_inherited_sentinel_as_confingo_owned():
    # The base's planted __hash__ resolves through the MRO while the subclass is
    # being created, so the guard has to recognize it rather than report every
    # derived node as hand-writing a hash.
    assert SentinelDerived.__dict__["__hash__"] is _unhashable_config
    built = from_dict(SentinelDerived, {"x": 1, "y": 2})
    assert (built.x, built.y) == (1, 2)
    assert SentinelDerived.__dict__["__hash__"] is None


def test_node_subclass_declared_after_the_base_is_touched_is_accepted():
    from_dict(TouchedBase, {})
    assert TouchedBase.__dict__["__hash__"] is None

    @dataclass
    class LateDerived(TouchedBase):
        pass

    assert LateDerived.__dict__["__hash__"] is _unhashable_config


# --- conflicting @dataclass flags ---------------------------------------------


def test_order_flag_rejected():
    with pytest.raises(ConfigError, match=r"order=True"):
        from_dict(RootWithOrderSection, {})


def test_eq_false_flag_rejected():
    with pytest.raises(ConfigError, match=r"eq=False"):
        from_dict(RootWithNoEqSection, {})


def test_unsafe_hash_flag_rejected():
    with pytest.raises(ConfigError, match=r"unsafe_hash=True"):
        from_dict(RootWithUnsafeHashSection, {})


def test_init_false_class_flag_rejected():
    with pytest.raises(ConfigError, match=r"init=False"):
        from_dict(RootWithNoInitSection, {})


def test_multiple_bad_flags_report_together():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(RootWithMultiFlagSection, {})
    message = str(excinfo.value)
    assert "order=True" in message
    assert "unsafe_hash=True" in message
    assert len(excinfo.value.issues) == 2


def test_inherited_eq_false_flag_rejected_via_mro():
    with pytest.raises(ConfigError, match=r"eq=False"):
        from_dict(RootInheritsEqFalse, {})


# --- repeated touches (cache ordering) ----------------------------------------


def test_bad_class_rejected_on_every_touch():
    with pytest.raises(ConfigError, match="custom __eq__"):
        from_dict(RootWithCustomEqSection, {})
    with pytest.raises(ConfigError, match="custom __eq__"):
        from_dict(RootWithCustomEqSection, {})


def test_good_class_builds_repeatedly():
    first = from_dict(GoodRoot, {})
    second = from_dict(GoodRoot, {})
    assert config_equal(first, second)


# --- frozen sections are unhashable -------------------------------------------


def test_frozen_section_hash_disabled():
    built = from_dict(GoodRoot, {})
    assert type(built.fs).__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built.fs)
    assert to_dict(built) == {"fs": {"a": 0, "b": 1.5}, "plain": 3}
    assert config_equal(GoodRoot(), GoodRoot())
    assert config_hash(GoodRoot()) == config_hash(GoodRoot())


def test_two_equal_frozen_sections_compare_equal():
    assert FrozenSection(1, 2.0) == FrozenSection(1, 2.0)
    assert FrozenSection(1, 2.0) != FrozenSection(2, 2.0)


def test_frozen_section_with_array_is_unhashable_rather_than_raising_on_the_array():
    built = from_dict(RootWithFrozenArray, {})
    assert type(built.fa).__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built.fa)
    assert config_hash(built.fa) == config_hash(built.fa)


# --- slots --------------------------------------------------------------------


def test_slots_root_and_section_build_and_round_trip():
    built = from_dict(SlotsRoot, {"inner": {"lr": 0.5}})
    assert built.inner.derived == 5.0
    assert to_dict(built) == {"inner": {"lr": 0.5, "steps": 10}, "name": "run"}
    assert config_equal(built, from_dict(SlotsRoot, {"inner": {"lr": 0.5}}))
    assert config_hash(built) == config_hash(from_dict(SlotsRoot, {"inner": {"lr": 0.5}}))


def test_slots_completeness_check_fires_on_unset_init_false_field():
    with pytest.raises(ConfigError, match="missing: init=False field was not set"):
        from_dict(SlotsIncompleteRoot, {"v": 1})


def test_slotted_plain_section_is_unhashable_after_first_touch():
    from_dict(SlotsRoot, {"inner": {"lr": 0.5}})
    assert SlotsInner.__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(SlotsInner(lr=0.5))


def test_slots_with_weakref_slot_accepted():
    built = from_dict(SlotsWeakrefRoot, {"x": 5})
    assert built.x == 5
    assert SlotsWeakrefRoot(1) == SlotsWeakrefRoot(1)
    assert SlotsWeakrefRoot.__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built)


# --- generated methods are not misflagged -------------------------------------


def test_generated_eq_is_installed_not_rejected():
    built = from_dict(GoodRoot, {})
    assert type(built).__eq__.__qualname__ == "_canonical_eq"
    assert type(built.fs).__eq__.__qualname__ == "_canonical_eq"


# --- inherited hashes shadowed on undecorated subclasses -----------------------


@dataclass
class MutableHashBase:
    a: int = 0


class InheritsMutableHash(MutableHashBase):
    pass


@dataclass
class RootInheritsMutableHash(ConfigNode):
    sec: InheritsMutableHash = field(default_factory=InheritsMutableHash)  # pyrefly: ignore[bad-assignment]


@dataclass(frozen=True)
class FrozenHashBase:
    a: int = 0


class InheritsFrozenHash(FrozenHashBase):
    pass


@dataclass
class RootInheritsFrozenHash(ConfigNode):
    sec: InheritsFrozenHash = field(default_factory=InheritsFrozenHash)  # pyrefly: ignore[bad-assignment]


def test_mutable_undecorated_subclass_is_unhashable():
    built = from_dict(RootInheritsMutableHash, {})
    assert type(built.sec).__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built.sec)
    assert MutableHashBase.__dict__["__hash__"] is None


def test_frozen_undecorated_subclass_is_unhashable_and_leaves_its_base_alone():
    built = from_dict(RootInheritsFrozenHash, {})
    assert type(built.sec).__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built.sec)
    # The untouched base keeps the hash @dataclass(frozen=True) generated for it.
    assert FrozenHashBase.__dict__["__hash__"] is not None
    assert hash(FrozenHashBase()) == hash(FrozenHashBase())


# --- root flag interactions ---------------------------------------------------


@dataclass(order=True)
class OrderRoot(ConfigNode):
    x: int = 0


@dataclass(eq=False)
class EqFalseRoot(ConfigNode):
    x: int = 0


@dataclass(init=False)
class InitFalseRoot(ConfigNode):
    x: int = 0


def test_order_root_rejected_at_first_touch():
    with pytest.raises(ConfigError, match=r"order=True"):
        from_dict(OrderRoot, {})


def test_eq_false_root_rejected_at_first_touch():
    with pytest.raises(ConfigError, match=r"eq=False"):
        from_dict(EqFalseRoot, {})


def test_init_false_root_rejected_at_first_touch():
    with pytest.raises(ConfigError, match=r"init=False"):
        from_dict(InitFalseRoot, {})


def test_unsafe_hash_root_fails_at_class_creation():
    # The node's planted __hash__ is what a frozen subclass needs to stay
    # unhashable, and it is also what the decorator refuses to overwrite, so this
    # one flag reports through the standard library rather than through confingo's
    # preflight guard.
    with pytest.raises(TypeError, match="Cannot overwrite attribute __hash__"):

        @dataclass(unsafe_hash=True)
        class UnsafeRoot(ConfigNode):
            x: int = 0


def test_root_body_eq_and_hash_reported_together():
    with pytest.raises(ConfigError) as excinfo:

        @dataclass
        class BothRoot(ConfigNode):
            x: int = 0

            def __eq__(self, other: object) -> bool:  # pyrefly: ignore[missing-override-decorator]
                return True

            def __hash__(self) -> int:
                return 1

    assert len(excinfo.value.issues) == 2
