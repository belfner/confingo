"""Pending lifecycle reporting: which paths a failed build names for the next load.

A build reaches a node's lifecycle after its fields build, so a field issue leaves
``__post_init__``, the ``init=False`` completeness check, and ``__validate__``
ahead of it. ``ConfigError.pending_lifecycle_paths`` names where that work waits,
and these tests pin which paths appear, in which order, and where the reading
stays quiet.
"""

from __future__ import annotations

from collections import UserList
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import pytest

from confingo import ConfigError
from confingo._errors import _IssueCollector
from confingo._schema import MAX_PLAIN_DEPTH
from confingo.functional import from_dict


@dataclass
class Leaf:
    """The deepest node of the nested fixture, declaring an invariant."""

    lr: float = 0.1

    def __validate__(self) -> list[str]:
        """Report the invariant this leaf owns.

        Returns:
          list[str]: One message when the rate is unusable.
        """
        return [] if self.lr > 0 else ["lr must be positive"]


@dataclass
class Middle:
    """The intermediate node, declaring an invariant of its own."""

    leaf: Leaf
    scale: int = 2

    def __validate__(self) -> list[str]:
        """Report the invariant this node owns.

        Returns:
          list[str]: One message when the scale is unusable.
        """
        return [] if self.scale > 0 else ["scale must be positive"]


@dataclass
class Nested:
    """The root of the nested fixture."""

    middle: Middle
    epochs: int = 1

    def __validate__(self) -> list[str]:
        """Report the invariant this root owns.

        Returns:
          list[str]: One message when the epoch count is unusable.
        """
        return [] if self.epochs > 0 else ["epochs must be positive"]


def _pending(config_cls: type[Any], data: dict[str, Any]) -> tuple[str, ...]:
    """Build a config expecting failure and read the paths it left pending.

    Args:
      config_cls (type[Any]): The schema to build.
      data (dict[str, Any]): The mapping to build from.

    Returns:
      tuple[str, ...]: The pending lifecycle paths the failure carried.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, data)
    return info.value.pending_lifecycle_paths


def test_blocked_chain_names_every_ancestor_in_discovery_order():
    broken = {"epochs": -1, "middle": {"scale": -1, "leaf": {"lr": "fast"}}}
    with pytest.raises(ConfigError) as info:
        from_dict(Nested, broken)
    error = info.value
    assert error.pending_lifecycle_paths == ("middle.leaf", "middle", "")
    # The invariants stay ahead of this load, so only the field issue reports.
    assert [str(issue) for issue in error.issues] == ["middle.leaf.lr: expected float, got str"]


def test_repaired_types_release_every_invariant_at_once():
    repaired = {"epochs": -1, "middle": {"scale": -1, "leaf": {"lr": -0.5}}}
    with pytest.raises(ConfigError) as info:
        from_dict(Nested, repaired)
    error = info.value
    assert [str(issue) for issue in error.issues] == [
        "middle.leaf: lr must be positive",
        "middle: scale must be positive",
        "<root>: epochs must be positive",
    ]
    # Every node reached its hook, so the tuple is empty.
    assert error.pending_lifecycle_paths == ()
    assert "Pending lifecycle work" not in str(error)


@dataclass
class PlainLeaf:
    """A leaf section, declaring one field."""

    n: int = 0


@dataclass
class PlainRoot:
    """A root whose tree declares fields alone."""

    leaf: PlainLeaf


def test_hook_free_tree_stays_quiet():
    with pytest.raises(ConfigError) as info:
        from_dict(PlainRoot, {"leaf": {"n": "bad"}})
    error = info.value
    assert error.pending_lifecycle_paths == ()
    assert str(error) == "config has 1 issue:\n  - leaf.n: expected int, got str"


@dataclass
class Hooked:
    """A section declaring an invariant, for the pre-construction exits."""

    n: int = 0

    def __validate__(self) -> list[str]:
        """Answer with an empty report; these tests read the pending path.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class HoldsHooked:
    """A root holding one hooked section."""

    section: Hooked


def test_non_mapping_section_records_its_lifecycle():
    assert _pending(HoldsHooked, {"section": "not-a-mapping"}) == ("section",)


def test_self_referential_mapping_records_its_lifecycle():
    data: dict[str, Any] = {}
    data["section"] = data
    assert _pending(HoldsHooked, data) == ("section",)


@dataclass
class RaisingInit:
    """A section whose hand-written constructor raises before ``__post_init__``."""

    x: int = 0

    def __init__(self, x: int = 0) -> None:
        if x < 0:
            raise ValueError("x must be non-negative")
        self.x = x
        self.__post_init__()

    def __post_init__(self) -> None:
        """Stand in for the work a repaired load reaches."""
        return


@dataclass
class HoldsRaisingInit:
    """A root holding the raising-constructor section."""

    node: RaisingInit


def test_constructor_raising_before_post_init_keeps_post_init_pending():
    # The call names one step from outside, so a constructor that raised before
    # __post_init__ reads the same as one that raised inside it; the hook stays
    # pending either way, since it runs again once the issue is fixed.
    assert _pending(HoldsRaisingInit, {"node": {"x": -1}}) == ("node",)


@dataclass
class ValidatesAfterUnset:
    """A section whose completeness check fails ahead of its invariant."""

    a: int = 0
    b: int = field(init=False)

    def __post_init__(self) -> None:
        """Leave ``b`` for the completeness check to find."""
        return

    def __validate__(self) -> list[str]:
        """Report the invariant a later load reaches.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class HoldsUnset:
    """A root holding the incomplete section."""

    node: ValidatesAfterUnset


def test_completeness_failure_leaves_validate_pending():
    assert _pending(HoldsUnset, {"node": {"a": 1}}) == ("node",)


@dataclass
class UnsetWithoutValidate:
    """A section whose completeness fails with every later stage behind it."""

    a: int = 0
    b: int = field(init=False)

    def __post_init__(self) -> None:
        """Leave ``b`` for the completeness check to find."""
        return


@dataclass
class HoldsUnsetWithoutValidate:
    """A root holding the section whose later stages are all behind it."""

    node: UnsetWithoutValidate


def test_completeness_failure_records_nothing_when_validate_is_absent():
    # Construction and the completeness check are behind this node, and its
    # declarations stop at completeness, so the stage flags leave it out. A call
    # that left post_init or completeness enabled would name it here, since the
    # node declares both.
    assert _pending(HoldsUnsetWithoutValidate, {"node": {"a": 1}}) == ()


@dataclass
class PostInitOnly:
    """A section whose only lifecycle work is a post-init hook."""

    a: int = 0

    def __post_init__(self) -> None:
        """Stand in for the work a repaired load reaches."""
        return


@dataclass
class HoldsPostInitOnly:
    """A root holding the post-init-only section."""

    node: PostInitOnly


def test_post_init_alone_is_enough_to_record_a_blocked_node():
    # A node whose one declaration is a post-init hook still carries work the
    # next load reaches, which independently pins the post-init stage.
    assert _pending(HoldsPostInitOnly, {"node": {"a": "bad"}}) == ("node",)


@dataclass
class UnsetOnly:
    """A section whose only lifecycle work is its completeness check."""

    a: int = 0
    b: int = field(init=False, default=0)


@dataclass
class HoldsUnsetOnly:
    """A root holding the completeness-only section."""

    node: UnsetOnly


def test_completeness_alone_is_enough_to_record_a_blocked_node():
    assert _pending(HoldsUnsetOnly, {"node": {"a": "bad"}}) == ("node",)


@dataclass
class HookedChild:
    """A child section declaring an invariant."""

    n: int = 1

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the hook's presence is what is read.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class HookFreeHead:
    """An authored head whose declarations stop at post-init, holding a hooked child."""

    child: HookedChild = field(default_factory=HookedChild)
    runtime: str = field(init=False)

    def __post_init__(self) -> None:
        """Leave ``runtime`` for the completeness check to find."""
        return


@dataclass
class HoldsHookFreeHead:
    """A root selecting the hook-free head through a factory."""

    head: HookFreeHead = field(default_factory=HookFreeHead)


def test_authored_barrier_records_when_only_a_descendant_carries_the_hook():
    # The walk stops at the head, so the child's hook stays unvisited. One entry
    # names the barrier, covering what sits beneath it.
    assert _pending(HoldsHookFreeHead, {}) == ("head",)


@dataclass
class FactorySection:
    """The section an invalid factory fails to produce."""

    x: int = -1

    def __validate__(self) -> list[str]:
        """Report the invariant a repaired factory reaches.

        Returns:
          list[str]: One message when the value is unusable.
        """
        return ["x must be positive"] if self.x <= 0 else []


@dataclass
class ScalarProductForSection:
    """A root whose factory produces a leaf where a section is declared."""

    section: FactorySection = field(default_factory=lambda: "bad")  # type: ignore[arg-type, return-value]


def test_scalar_product_under_a_section_annotation_records_the_barrier():
    # The product is a leaf precisely because it missed the annotation, so the
    # annotation is what answers for the section a repaired factory builds.
    assert _pending(ScalarProductForSection, {}) == ("section",)


@dataclass
class ScalarProductForScalar:
    """A root whose factory produces a leaf where a leaf is declared."""

    n: int = field(default_factory=lambda: "bad")  # type: ignore[arg-type, return-value]


def test_scalar_product_under_a_scalar_annotation_stays_quiet():
    assert _pending(ScalarProductForScalar, {}) == ()


@dataclass
class Deep:
    """A self-referential section, for reaching the plain-depth budget."""

    child: Deep | None = None

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the pending path is what this test reads.

        Returns:
          list[str]: Always empty.
        """
        return []


def test_depth_budget_records_the_section_it_stops_at():
    data: dict[str, Any] = {}
    node = data
    for _ in range(70):
        child: dict[str, Any] = {}
        node["child"] = child
        node = child
    pending = _pending(Deep, data)
    # The section the budget stops at is the deepest entry, and every ancestor
    # follows it up to the root. Naming that first path exactly is what separates
    # the depth site's own record from the ancestor records that a node-failure
    # gate would produce anyway.
    assert pending[0] == ".".join(["child"] * MAX_PLAIN_DEPTH)
    assert pending[-1] == ""
    assert len(pending) == MAX_PLAIN_DEPTH + 1
    assert list(pending) == [".".join(["child"] * level) for level in range(MAX_PLAIN_DEPTH, -1, -1)]


def _quiet_hook() -> list[str]:
    """Answer the hook contract with an empty report.

    Returns:
      list[str]: Always empty.
    """
    return []


class _Descriptor:
    """A non-callable binding that supplies a callable once an instance exists."""

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        """Answer with a hook bound to the instance.

        Args:
          instance (Any): The instance the attribute was read from.
          owner (type[Any] | None = None): The class that owns the binding.

        Returns:
          Any: A callable answering the hook contract.
        """
        return _quiet_hook


class _HostileMeta(type):
    """A metaclass that raises on ordinary attribute access."""

    def __getattribute__(cls, name: str) -> Any:
        """Raise for every read, standing in for author code that misbehaves.

        Args:
          name (str): The attribute being read.

        Raises:
          RuntimeError: On every read.
        """
        raise RuntimeError("hostile metaclass")


class _HostileDescriptorType(metaclass=_HostileMeta):
    """A descriptor whose own type answers reads with a raise."""


@dataclass
class DescriptorHook(Hooked):
    """A section binding its invariant through a descriptor."""

    __validate__ = _Descriptor()  # type: ignore[assignment]


@dataclass
class HostileBinding(Hooked):
    """A section whose invariant binding has a hostile type."""

    __validate__ = _HostileDescriptorType()  # type: ignore[assignment]


@dataclass
class HoldsDescriptorHook:
    """A root holding the descriptor-bound section."""

    section: DescriptorHook


@dataclass
class HoldsHostileBinding:
    """A root holding the hostile-typed binding."""

    section: HostileBinding


def test_descriptor_binding_counts_as_a_declared_hook():
    assert _pending(HoldsDescriptorHook, {"section": "not-a-mapping"}) == ("section",)


def test_hostile_binding_type_still_reports_the_config():
    # Reading the binding's type through raw namespaces leaves the probe inside
    # those namespaces, so the config's own issue is what arrives.
    with pytest.raises(ConfigError) as info:
        from_dict(HoldsHostileBinding, {"section": "not-a-mapping"})
    error = info.value
    assert "expected a mapping" in str(error.issues[0])
    # The raw scan classifies the binding as a plain value, so the section reads
    # as one whose hook is shadowed, giving an empty tuple. Asserting it separates
    # the classification from the RuntimeError question.
    assert error.pending_lifecycle_paths == ()


@dataclass
class BuiltButInvalid:
    """A section a factory constructs with a value its annotation declines."""

    n: int = 0

    def __validate__(self) -> list[str]:
        """Report the invariant a repaired factory reaches.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class HoldsInvalidProduct:
    """A root whose factory builds a section carrying a wrong leaf type."""

    section: BuiltButInvalid = field(default_factory=lambda: BuiltButInvalid(n="bad"))  # type: ignore[arg-type]


def test_constructed_invalid_product_records_its_barrier():
    # The product is a real section, so the product arm answers even before the
    # annotation is consulted.
    assert _pending(HoldsInvalidProduct, {}) == ("section",)


@dataclass
class CustomSequenceRoot:
    """A root whose factory answers a leaf field with a custom sequence.

    The product is a ``UserList``, which the ``Sequence`` reading covers, so it
    reaches the product arm when that arm reads ``Sequence``. The annotation is a
    leaf, so the product arm alone answers.
    """

    n: int = field(default_factory=lambda: UserList([BuiltButInvalid()]))  # type: ignore[arg-type, assignment]


def test_custom_sequence_product_records_its_barrier():
    # A UserList is a Sequence carrying a real section, so the product arm answers
    # while the declared hint stays a leaf.
    assert _pending(CustomSequenceRoot, {}) == ("n",)


@dataclass
class StringProductRoot:
    """A root whose factory answers a leaf field with a string."""

    n: int = field(default_factory=lambda: "text")  # type: ignore[arg-type, assignment]


def test_string_product_stays_a_leaf():
    # str is a Sequence of itself, and the leaf reading is what answers here.
    assert _pending(StringProductRoot, {}) == ()


@dataclass
class RaisingFactoryRoot:
    """A root whose factory raises before any product exists."""

    section: BuiltButInvalid = field(default_factory=lambda: (_ for _ in ()).throw(ValueError("boom")))


def test_raising_factory_records_the_barrier_from_the_annotation():
    # The factory raised ahead of making a product, so the annotation is the
    # whole reading.
    assert _pending(RaisingFactoryRoot, {}) == ("section",)


@dataclass
class Inherits(Hooked):
    """A section inheriting its invariant from a base."""


@dataclass
class Shadows(Hooked):
    """A section whose own binding shadows the inherited invariant."""

    __validate__ = None  # type: ignore[assignment]


@dataclass
class NonCallable(Hooked):
    """A section binding a value the lifecycle would decline to call."""

    __validate__ = "text"  # type: ignore[assignment]


@dataclass
class HoldsInherits:
    """A root holding the inheriting section."""

    section: Inherits


@dataclass
class HoldsShadows:
    """A root holding the shadowing section."""

    section: Shadows


@dataclass
class HoldsNonCallable:
    """A root holding the non-callable binding."""

    section: NonCallable


@pytest.mark.parametrize(
    ("config_cls", "expected"),
    [
        (HoldsInherits, ("section",)),
        (HoldsShadows, ()),
        (HoldsNonCallable, ()),
    ],
)
def test_hook_detection_reads_the_nearest_binding(config_cls: type[Any], expected: tuple[str, ...]):
    assert _pending(config_cls, {"section": "not-a-mapping"}) == expected


@dataclass
class AlphaInner:
    """The section reached under the first union member."""

    a: int

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the path this section contributes is read.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class BetaInner:
    """The section reached under the second union member."""

    b: int

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the path this section contributes is read.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class AlphaMember:
    """A union member holding its own nested section."""

    inner: AlphaInner

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the path this member contributes is read.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class BetaMember:
    """A union member holding a differently named nested section."""

    other: BetaInner

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the path this member contributes is read.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class UnionRoot:
    """A root whose one field names two lists of sections.

    Each member descends into sections of its own, which is what makes both
    trials record pending paths and the adopted ones say which trial was taken.
    """

    choice: list[AlphaMember] | list[BetaMember]


def test_union_adopts_only_the_best_failed_member():
    # Each member reaches a differently named nested section, so the adopted paths
    # say which trial was taken; equal paths would hide a leak behind dedup.
    pending = _pending(UnionRoot, {"choice": [{"inner": {"a": "bad"}}]})
    assert pending == ("choice.0.inner", "choice.0")
    assert "choice.0.other" not in pending


def test_pending_paths_stay_out_of_member_selection():
    # The first member fails and records pending paths while it is tried. The
    # second converts cleanly and is what builds, which is the property that keeps
    # a diagnostic from deciding which member a config gets.
    built = from_dict(UnionRoot, {"choice": [{"other": {"b": 2}}]})
    assert isinstance(built.choice[0], BetaMember)
    assert built.choice[0].other.b == 2


def test_pending_paths_leave_a_collector_clean():
    # The load-bearing rule, read directly: a collector holding pending paths and
    # an empty issue list is clean, which is what a union member's trial is tested
    # on. Reading pending paths in clean() would make this False and hand member
    # selection to a diagnostic.
    collector = _IssueCollector()
    collector.add_pending_lifecycle("section")
    collector.add_pending_lifecycle("")
    assert collector.clean() is True
    assert collector.pending_lifecycle_paths == ["section", ""]


def test_pending_paths_stay_out_of_the_best_match_count():
    # Two failed members with one issue each: the tie goes to the first declared
    # member, TiedAlpha. TiedAlpha is also the one recording a pending path, so a
    # ranking that summed issues and pending paths would score it 2 against
    # TiedBeta's 1 and name TiedBeta instead.
    with pytest.raises(ConfigError) as info:
        from_dict(TiedUnionRoot, {"choice": [{"a": "bad", "shared": 1}]})
    assert "best match list[TiedAlpha]" in str(info.value.issues[0])


def test_collector_collapses_a_repeated_path():
    collector = _IssueCollector()
    collector.add_pending_lifecycle("section")
    collector.extend_pending_lifecycle(["section", "other", "section"])
    assert collector.pending_lifecycle_paths == ["section", "other"]


@dataclass
class TiedAlpha:
    """A union member carrying one issue and declaring an invariant.

    The invariant sits on the first member deliberately: a ranking that counted
    pending paths would score this member worse than its quiet sibling and hand
    the report to the sibling, which is what the tie test detects.
    """

    a: int
    shared: int = 0

    def __validate__(self) -> list[str]:
        """Answer with an empty report; the pending path it records is the point.

        Returns:
          list[str]: Always empty.
        """
        return []


@dataclass
class TiedBeta:
    """A union member carrying one issue and declaring fields alone."""

    a: int
    shared: int = 0


@dataclass
class TiedUnionRoot:
    """A root whose members fail with an equal issue count."""

    choice: list[TiedAlpha] | list[TiedBeta]


@dataclass
class WideRoot:
    """A root with several blocked siblings, for order and rendering."""

    one: Hooked
    two: Hooked
    three: Hooked
    four: Hooked
    five: Hooked
    six: Hooked


def test_siblings_keep_walk_order_and_render_capped():
    blocked = dict.fromkeys(("one", "two", "three", "four", "five", "six"), "not-a-mapping")
    with pytest.raises(ConfigError) as info:
        from_dict(WideRoot, blocked)
    error = info.value
    assert error.pending_lifecycle_paths == ("one", "two", "three", "four", "five", "six")
    # The attribute carries every path; the sentence names five and counts the rest.
    assert "at one, two, three, four, five and 1 more:" in str(error)
    assert str(error).count("Pending lifecycle work") == 1


def test_root_renders_with_its_label():
    broken = {"epochs": -1, "middle": {"scale": -1, "leaf": {"lr": "fast"}}}
    with pytest.raises(ConfigError) as info:
        from_dict(Nested, broken)
    rendered = str(info.value)
    assert "middle.leaf, middle, <root>" in rendered
    # The pending line follows the issue list and leaves the count alone.
    assert rendered.index("Pending lifecycle work") > rendered.index("middle.leaf.lr")
    assert rendered.startswith("config has 1 issue:")


def test_direct_construction_defaults_to_no_pending_paths():
    error = ConfigError.single("boom", context="config")
    assert error.pending_lifecycle_paths == ()
    assert "Pending lifecycle work" not in str(error)


def test_pending_paths_freeze_to_a_tuple():
    supplied = ["a", "b"]
    error = ConfigError([], context="config", pending_lifecycle_paths=supplied)
    supplied.append("c")
    assert error.pending_lifecycle_paths == ("a", "b")
