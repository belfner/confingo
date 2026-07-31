"""Tests for variant groups: one annotation standing for a closed set of sections.

A group is declared by subclassing ``ConfigChoice`` with the key its sections
select under; each variant subclasses the group with the selection string a
config file writes under that key. The annotation names the group, the file names
the variant, and exactly one section is built.
"""

from __future__ import annotations

import weakref
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
    ConfigChoice,
    ConfigError,
    ConfigNode,
    _schema,
)
from confingo._schema import (
    MAX_PLAIN_DEPTH,
    plain_cycle_message,
    plain_depth_message,
)
from confingo.functional import (
    config_hash,
    from_dict,
    to_dict,
    validate_schema,
)


if TYPE_CHECKING:
    from pathlib import Path


HOOK_CALLS: list[str] = []
"""Construction-hook call log, appended to by the variants below."""


@dataclass
class Optimizer(ConfigChoice, tag_key="algorithm"):
    """Variant group whose members share a learning rate."""

    lr: float = 1e-3


@dataclass
class AdamW(Optimizer, tag="adamw"):
    """Variant carrying moment coefficients."""

    betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        HOOK_CALLS.append("AdamW")


@dataclass
class SGD(Optimizer, tag="sgd"):
    """Variant carrying a momentum term."""

    momentum: float = 0.9

    def __post_init__(self) -> None:
        HOOK_CALLS.append("SGD")

    def __validate__(self):
        if self.momentum < 0.0:
            yield f"momentum must be non-negative, got {self.momentum}"


@dataclass
class Train(ConfigNode):
    """Root selecting one optimizer."""

    optimizer: Optimizer
    workers: int = 8


def _issues(config_cls: type[Any], data: dict[str, Any]) -> list[tuple[str, str]]:
    """Build ``config_cls`` from ``data`` and return its issues in report order.

    Args:
      config_cls (type[Any]): The schema class expected to fail.
      data (dict[str, Any]): The mapping to build from.

    Returns:
      list[tuple[str, str]]: One ``(path, message)`` pair per issue, in order.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, data)
    return [(issue.path, issue.message) for issue in info.value.issues]


# --- declaration --------------------------------------------------------------


def test_a_group_subclass_without_a_tag_is_rejected():
    with pytest.raises(ConfigError, match=r'tag="\.\.\."'):

        @dataclass
        class Untagged(Optimizer):
            pass


def test_a_selection_string_another_variant_carries_is_rejected():
    with pytest.raises(ConfigError, match="which AdamW already carries"):

        @dataclass
        class Rival(Optimizer, tag="adamw"):
            pass


def test_a_group_declared_inside_a_group_is_rejected():
    with pytest.raises(ConfigError, match="a variant is a leaf"):

        @dataclass
        class Inner(Optimizer, tag_key="nested"):
            pass


def test_a_config_choice_subclass_without_a_key_is_rejected():
    with pytest.raises(ConfigError, match=r'tag_key="\.\.\."'):

        @dataclass
        class Keyless(ConfigChoice):
            pass


def test_declaring_both_keywords_is_rejected():
    with pytest.raises(ConfigError, match="declares both tag_key= and tag="):

        @dataclass
        class Both(ConfigChoice, tag_key="k", tag="t"):
            pass


def test_a_tag_that_is_not_a_non_empty_string_is_rejected():
    with pytest.raises(ConfigError, match="write a non-empty str"):

        @dataclass
        class Numbered(Optimizer, tag=3):
            pass


def test_a_tag_carrying_a_str_subclass_is_rejected():
    # A file carries the exact class str, so a tag of another class writes a
    # section that reads back as a plain str and compares unequal to the tag
    # that wrote it, which reports 'one' against 'one'.
    class Tag(str):
        pass

    with pytest.raises(ConfigError, match="A file carries the exact class str"):

        @dataclass
        class Subclassed(Optimizer, tag=Tag("subclassed")):
            pass


def test_a_group_key_carrying_a_str_subclass_is_rejected():
    class Key(str):
        pass

    with pytest.raises(ConfigError, match="A file carries the exact class str"):

        @dataclass
        class Subkeyed(ConfigChoice, tag_key=Key("kind")):
            pass


@dataclass
class Righthand(ConfigChoice, tag_key="side"):
    """A second group, whose variant another group's subclass tries to inherit."""


@dataclass
class RighthandOne(Righthand, tag="one"):
    """A registered variant of the second group."""

    r: int = 1


def test_a_class_inheriting_a_variant_of_another_group_is_rejected():
    # A variant subclasses its own group, so inheriting one brings that group
    # into the MRO too: the class would register under one string while
    # answering to two lineages, and it is named by the pair it descends from.
    with pytest.raises(ConfigError, match="descends from the variant groups Leafy, Righthand"):

        @dataclass
        class Crossed(Leafy, RighthandOne, tag="crossed"):
            pass


def test_a_class_descending_from_two_groups_is_rejected():
    with pytest.raises(ConfigError, match="descends from the variant groups"):

        @dataclass
        class Sided(Leafy, Righthand, tag="sided"):
            pass


# --- dispatch -----------------------------------------------------------------


def test_the_selection_string_names_the_variant_that_is_built():
    built = from_dict(Train, {"optimizer": {"algorithm": "sgd", "lr": 0.1, "momentum": 0.8}})
    assert built.optimizer == SGD(lr=0.1, momentum=0.8)


def test_a_variant_inherits_the_fields_the_group_declares():
    built = from_dict(Train, {"optimizer": {"algorithm": "adamw"}})
    assert built.optimizer.lr == 1e-3


def test_a_group_section_round_trips_through_the_enclosing_field():
    built = from_dict(Train, {"optimizer": {"algorithm": "sgd", "momentum": 0.8}})
    assert to_dict(built)["optimizer"] == {"algorithm": "sgd", "lr": 1e-3, "momentum": 0.8}
    assert from_dict(Train, to_dict(built)) == built


def test_the_selection_leads_the_section_it_describes():
    plain = to_dict(from_dict(Train, {"optimizer": {"algorithm": "sgd"}}))
    assert list(plain["optimizer"]) == ["algorithm", "lr", "momentum"]


def test_a_variant_marshalled_on_its_own_carries_its_selection():
    # The selection comes from the object's own class, so a variant rendered
    # outside the field it was reached through still reads back.
    assert to_dict(AdamW(lr=0.2)) == {"algorithm": "adamw", "lr": 0.2, "betas": [0.9, 0.999]}
    assert from_dict(AdamW, to_dict(AdamW(lr=0.2))) == AdamW(lr=0.2)


def test_a_group_entry_class_dispatches_on_the_file(tmp_path: Path):
    assert Optimizer.cfg.from_dict({"algorithm": "sgd", "momentum": 0.5}) == SGD(momentum=0.5)
    path = SGD(momentum=0.5).cfg.save_json(tmp_path / "opt.json")
    assert Optimizer.cfg.load_json(path) == SGD(momentum=0.5)


def test_a_variant_annotation_accepts_a_section_naming_that_variant():
    assert from_dict(AdamW, {"algorithm": "adamw", "lr": 0.5}) == AdamW(lr=0.5)
    assert from_dict(AdamW, {"lr": 0.5}) == AdamW(lr=0.5)


def test_a_variant_annotation_rejects_a_section_naming_another_variant():
    assert _issues(AdamW, {"algorithm": "sgd"}) == [
        (
            "algorithm",
            "expected 'adamw', the selection string AdamW carries, got 'sgd'; annotate the field with "
            "Optimizer to let a config file choose among 'adamw' | 'sgd'",
        )
    ]


def test_variants_with_equal_field_values_fingerprint_apart():
    assert config_hash(AdamW(lr=0.1)) != config_hash(SGD(lr=0.1))


def test_exactly_one_variant_is_constructed():
    HOOK_CALLS.clear()
    from_dict(Train, {"optimizer": {"algorithm": "sgd"}})
    # The selection names one class, so no other variant's constructor runs.
    assert HOOK_CALLS == ["SGD"]


# --- reporting ----------------------------------------------------------------


def test_an_omitted_group_section_reports_the_selection_at_its_own_path():
    assert _issues(Train, {}) == [("optimizer.algorithm", "missing required value (expected one of 'adamw' | 'sgd')")]


def test_a_section_without_a_selection_reports_the_selection_at_its_own_path():
    assert _issues(Train, {"optimizer": {"lr": 0.1}}) == [
        ("optimizer.algorithm", "missing required value (expected one of 'adamw' | 'sgd')")
    ]


def test_an_unknown_selection_names_every_registered_option():
    assert _issues(Train, {"optimizer": {"algorithm": "adam"}}) == [
        ("optimizer.algorithm", "expected one of 'adamw' | 'sgd', got 'adam'")
    ]


def test_a_group_section_is_a_mapping():
    assert _issues(Train, {"optimizer": "sgd"}) == [("optimizer", "expected a mapping for Optimizer, got str")]


def test_the_selection_key_is_not_reported_as_an_unknown_key():
    assert _issues(Train, {"optimizer": {"algorithm": "sgd", "nope": 1}}) == [
        ("optimizer.nope", "unknown key (known keys: lr, momentum)")
    ]


def test_a_selected_variant_reports_its_own_fields():
    assert _issues(Train, {"optimizer": {"algorithm": "sgd", "momentum": "fast"}}) == [
        ("optimizer.momentum", "expected float, got str")
    ]


def test_a_missing_selection_still_names_the_lifecycle_work_a_variant_declares():
    # No variant is selected, so any of them could be the one a fixed config
    # names; SGD declares __validate__, which is work the next load will run.
    with pytest.raises(ConfigError) as info:
        from_dict(Train, {"optimizer": {"lr": 0.1}})
    assert "optimizer" in info.value.pending_lifecycle_paths


# --- variants carrying identical fields ----------------------------------------


@dataclass
class Warmup(ConfigChoice, tag_key="schedule"):
    """Group whose variants declare the same fields as each other."""

    warmup_steps: int = 500
    total_steps: int = 10_000


@dataclass
class Cosine(Warmup, tag="cosine"):
    """One of two variants carrying no fields the other lacks."""


@dataclass
class Linear(Warmup, tag="linear"):
    """The second variant carrying no fields the other lacks."""


@dataclass
class Scheduled(ConfigNode):
    """Root selecting between two structurally identical variants."""

    schedule: Warmup


def test_variants_carrying_identical_fields_are_told_apart_by_the_selection():
    # Nothing in the fields distinguishes these two, so the key is the whole
    # answer, and it names one class rather than ranking two attempts.
    assert from_dict(Scheduled, {"schedule": {"schedule": "linear", "warmup_steps": 100}}).schedule == Linear(
        warmup_steps=100
    )
    assert from_dict(Scheduled, {"schedule": {"schedule": "cosine"}}).schedule == Cosine()


def test_a_typo_between_identical_variants_names_both_options():
    # A union of two identical shapes reported through whichever member came
    # closest, which a typo left tied. The key reports the whole set instead.
    assert _issues(Scheduled, {"schedule": {"schedule": "cosinus"}}) == [
        ("schedule.schedule", "expected one of 'cosine' | 'linear', got 'cosinus'")
    ]


# --- the union ban ------------------------------------------------------------


@dataclass
class LooseA(ConfigNode):
    """A section named beside another in one union."""

    x: int = 1


@dataclass
class LooseB(ConfigNode):
    """A second section named beside the first."""

    y: int = 2


def test_a_union_naming_two_sections_is_rejected():
    @dataclass
    class TwoSections(ConfigNode):
        choice: LooseA | LooseB = field(default_factory=LooseA)

    with pytest.raises(ConfigError, match="names 2 config sections in one union"):
        validate_schema(TwoSections)


def test_the_rejection_names_the_group_to_declare():
    @dataclass
    class TwoSections(ConfigNode):
        choice: LooseA | LooseB = field(default_factory=LooseA)

    with pytest.raises(ConfigError, match=r'class Group\(ConfigChoice, tag_key="\.\.\."\)'):
        validate_schema(TwoSections)


def test_the_ban_reaches_sections_inside_a_container():
    @dataclass
    class Nested(ConfigNode):
        items: list[LooseA | LooseB] = field(default_factory=list)

    with pytest.raises(ConfigError, match="names 2 config sections in one union"):
        validate_schema(Nested)


def test_a_group_counts_as_one_member_of_a_union():
    @dataclass
    class GroupBeside(ConfigNode):
        optimizer: Optimizer | int = 0

    validate_schema(GroupBeside)
    assert from_dict(GroupBeside, {"optimizer": 4}).optimizer == 4
    assert from_dict(GroupBeside, {"optimizer": {"algorithm": "sgd"}}).optimizer == SGD()


def test_one_section_beside_a_scalar_stays_legal():
    @dataclass
    class SectionBeside(ConfigNode):
        choice: LooseA | int = 0

    validate_schema(SectionBeside)


def test_an_optional_group_stays_legal():
    @dataclass
    class MaybeGroup(ConfigNode):
        optimizer: Optimizer | None = None

    validate_schema(MaybeGroup)
    assert from_dict(MaybeGroup, {"optimizer": None}).optimizer is None
    assert from_dict(MaybeGroup, {"optimizer": {"algorithm": "adamw"}}).optimizer == AdamW()


def test_a_set_of_group_sections_is_rejected():
    @dataclass
    class SetOfGroups(ConfigNode):
        optimizers: set[Optimizer] = field(default_factory=set)

    with pytest.raises(ConfigError, match="config sections are unhashable"):
        validate_schema(SetOfGroups)


# --- authored defaults --------------------------------------------------------


def test_a_factory_product_is_validated_field_by_field():
    # A group annotation admits any variant, so the class the product carries is
    # what its fields are checked against; an unchecked product would serialize
    # and fail only on the next load. An int under a float annotation is the case
    # a checker passes through, since the numeric tower promotes it, and the
    # value stays the int it was written as.
    @dataclass
    class BadFactory(ConfigNode):
        optimizer: Optimizer = field(default_factory=lambda: AdamW(lr=1))

    assert _issues(BadFactory, {}) == [
        (
            "optimizer.lr",
            "invalid default_factory value: expected a value already matching float, got int; "
            "defaults are validated as written",
        )
    ]


def test_a_valid_factory_product_is_passed_on_unchanged():
    @dataclass
    class GoodFactory(ConfigNode):
        optimizer: Optimizer = field(default_factory=lambda: AdamW(lr=0.5))

    assert from_dict(GoodFactory, {}).optimizer == AdamW(lr=0.5)


def test_a_factory_producing_the_group_base_is_rejected():
    @dataclass
    class BaseFactory(ConfigNode):
        optimizer: Optimizer = field(default_factory=Optimizer)

    reported = _issues(BaseFactory, {})
    assert reported[0][0] == "optimizer"
    assert "expected a variant of the group Optimizer" in reported[0][1]
    assert "build one of AdamW, SGD" in reported[0][1]


def test_marshalling_a_group_base_instance_is_rejected():
    with pytest.raises(ConfigError, match="build AdamW, SGD"):
        to_dict(Optimizer(lr=0.1))


# --- class recreation ---------------------------------------------------------


@dataclass
class Slotted(ConfigChoice, tag_key="kind"):
    """Group whose variant is rebuilt by the slots decorator."""


@dataclass(slots=True)
class SlottedVariant(Slotted, tag="slotted"):
    """Variant that @dataclass(slots=True) recreates after registration."""

    value: int = 1


@dataclass
class SlottedRoot(ConfigNode):
    """Root selecting the recreated variant."""

    section: Slotted


def test_a_slotted_variant_dispatches_through_the_surviving_class():
    # @dataclass(slots=True) discards the class __init_subclass__ first saw and
    # builds a replacement, so the registry has to name the class the module
    # kept rather than the one it threw away.
    assert "__slots__" in SlottedVariant.__dict__
    built = from_dict(SlottedRoot, {"section": {"kind": "slotted", "value": 5}})
    assert type(built.section) is SlottedVariant
    assert to_dict(built) == {"section": {"kind": "slotted", "value": 5}}


# --- preflight reaches the registry -------------------------------------------


@dataclass
class Reached(ConfigChoice, tag_key="kind"):
    """Group whose variant carries an annotation outside the supported set."""


@dataclass
class ReachedVariant(Reached, tag="broken"):
    """Variant reachable only through its group."""

    broken: complex = 0j


@dataclass
class ReachedRoot(ConfigNode):
    """Root naming the group alone."""

    section: Reached


def test_a_variants_schema_is_preflighted_through_the_group_annotation():
    # The group is the only class the annotation names, so a variant's own
    # annotations are reachable for checking through the registry alone.
    with pytest.raises(ConfigError, match="complex"):
        validate_schema(ReachedRoot)


@dataclass
class Late(ConfigChoice, tag_key="kind"):
    """Group gaining a variant after a root over it was already validated."""


@dataclass
class LateGood(Late, tag="good"):
    """The variant present at the first validation."""

    value: int = 1


@dataclass
class LateRoot(ConfigNode):
    """Root naming the group whose membership changes."""

    section: Late


def test_a_root_validated_before_a_variant_registers_is_validated_again():
    # The whole-tree result is cached on the entry class, and it depends on which
    # variants are registered, so a variant arriving later has to invalidate it.
    validate_schema(LateRoot)

    @dataclass
    class LateBroken(Late, tag="broken"):
        broken: complex = 0j

    with pytest.raises(ConfigError, match="complex"):
        validate_schema(LateRoot)


@dataclass
class Racing(ConfigChoice, tag_key="kind"):
    """Group gaining a variant while a walk over it is already running."""


@dataclass
class RacingGood(Racing, tag="good"):
    """The variant present when the walk starts."""

    value: int = 1


@dataclass
class RacingRoot(ConfigNode):
    """Root naming the group whose membership moves mid-walk."""

    section: Racing


def test_a_variant_registering_during_a_walk_is_not_cached_over(monkeypatch: pytest.MonkeyPatch):
    # Resolving an annotation can import a module, and that import can register a
    # variant the walk has already gone past. Storing the result under the
    # generation the walk opened with would then serve a clean answer for a
    # registry it never saw, and the very next build would dispatch into the
    # variant nothing checked.
    original = _schema._validate_dataclass_schema
    declared: list[type[Any]] = []

    def register_midway(config_cls: type[Any], *args: Any, **kwargs: Any) -> None:
        if config_cls is RacingGood and len(declared) == 0:

            @dataclass
            class RacingBroken(Racing, tag="broken"):
                broken: complex = 0j

            declared.append(RacingBroken)
        original(config_cls, *args, **kwargs)

    monkeypatch.setattr(_schema, "_validate_dataclass_schema", register_midway)
    with pytest.raises(ConfigError, match="complex"):
        validate_schema(RacingRoot)


@dataclass
class EmptyGroup(ConfigChoice, tag_key="kind"):
    """Group standing for nothing, so no config file can select a section."""


def test_a_group_with_no_variants_is_rejected():
    @dataclass
    class UsesEmpty(ConfigNode):
        section: EmptyGroup

    with pytest.raises(ConfigError, match="has no variants"):
        validate_schema(UsesEmpty)


@dataclass
class Colliding(ConfigChoice, tag_key="mode"):
    """Group whose key a variant also declares as a field."""


@dataclass
class CollidingVariant(Colliding, tag="one"):
    """Variant declaring a field named for its group's selection key."""

    mode: str = "shadow"


def test_a_field_named_for_the_selection_key_is_rejected():
    @dataclass
    class UsesColliding(ConfigNode):
        section: Colliding

    with pytest.raises(ConfigError, match="one key in the section would name both"):
        validate_schema(UsesColliding)


def test_a_variant_named_directly_is_held_to_the_selection_key():
    # The field names the variant rather than the group, so the collision is
    # reached through the variant's own walk. A field of that name would
    # overwrite the selection the marshal writes, leaving a section that rejects
    # the output it just produced.
    with pytest.raises(ConfigError, match="one key in the section would name both"):
        validate_schema(CollidingVariant)


# --- the entry class carries the group's own checks ----------------------------


def test_a_group_entry_class_is_preflighted_as_a_group():
    with pytest.raises(ConfigError, match="has no variants"):
        validate_schema(EmptyGroup)


def test_a_group_entry_class_preflights_its_variants():
    with pytest.raises(ConfigError, match="complex"):
        validate_schema(Reached)


def test_a_group_entry_build_preflights_its_variants():
    # Group.cfg.from_dict is a documented entry route, so it runs the same
    # preflight a field annotated with the group runs.
    with pytest.raises(ConfigError, match="complex"):
        Reached.cfg.from_dict({"kind": "broken"})


# --- one class per selection string --------------------------------------------


@dataclass
class Repeated(ConfigChoice, tag_key="kind"):
    """Group a factory declares variants for more than once."""


def _declare_repeated() -> type[Any]:
    """Declare a variant of ``Repeated`` under a fixed selection string.

    Returns:
      type[Any]: The freshly created variant class.
    """

    @dataclass
    class Made(Repeated, tag="made"):
        value: int = 1

    return Made


def test_a_second_class_claiming_a_taken_string_is_rejected():
    # Two calls produce two distinct classes carrying one qualified name. Letting
    # the second take the entry over would leave instances of the first with no
    # selection, so to_dict would drop it and the section would stop round
    # tripping. Only a recreation of the registered class, which arrives with no
    # tag= of its own, takes the entry.
    first = _declare_repeated()
    with pytest.raises(ConfigError, match="already carries"):
        _declare_repeated()
    assert to_dict(first(value=2)) == {"kind": "made", "value": 2}


@dataclass
class Leafy(ConfigChoice, tag_key="kind"):
    """Group whose variants are leaves."""


@dataclass
class LeafyOne(Leafy, tag="one"):
    """A variant another class tries to extend."""

    a: int = 1


def test_a_variant_extending_a_variant_is_rejected():
    with pytest.raises(ConfigError, match="already a variant of the group"):

        @dataclass
        class LeafyTwo(LeafyOne, tag="two"):
            b: int = 2


# --- collect-all across variants -----------------------------------------------


@dataclass
class Twins(ConfigChoice, tag_key="kind"):
    """Group whose variants repeat a shared field and each declare a defect."""

    shared: int = 1


@dataclass
class TwinA(Twins, tag="a"):
    """One variant carrying a defect of its own.

    An int under a float annotation is the case a checker passes through, since
    the numeric tower promotes it, and the value stays the int it was written as.
    """

    value: float = 1


@dataclass
class TwinB(Twins, tag="b"):
    """A second variant carrying an identical defect of its own."""

    value: float = 1


def test_each_variants_own_defect_is_reported():
    # The two defects share a path and a message, and they are two classes to
    # fix. Collapsing them would report one, and fixing it would reveal the other
    # on the next run.
    @dataclass
    class UsesTwins(ConfigNode):
        section: Twins

    with pytest.raises(ConfigError) as info:
        validate_schema(UsesTwins)
    defects = [issue for issue in info.value.issues if issue.path == "section.value"]
    assert len(defects) == 2


@dataclass
class SharedDefect(ConfigChoice, tag_key="kind"):
    """Group declaring a field every variant inherits."""

    broken: complex = 0j


@dataclass
class SharedOne(SharedDefect, tag="one"):
    """One variant inheriting the group's field."""


@dataclass
class SharedTwo(SharedDefect, tag="two"):
    """A second variant inheriting the same field."""


@dataclass
class UsesShared(ConfigNode):
    """Root naming the group whose own field is defective."""

    section: SharedDefect


@dataclass
class Overridden(ConfigChoice, tag_key="kind"):
    """Group whose defective field a variant re-declares identically."""

    value: float = 1


@dataclass
class OverriddenOne(Overridden, tag="one"):
    """Variant restating the group's defective declaration as its own."""

    value: float = 1


@dataclass
class UsesOverridden(ConfigNode):
    """Root naming the group whose field is declared twice."""

    section: Overridden


def test_a_variant_redeclaring_a_defective_field_is_reported_separately():
    # Two declarations produce one path and one message, and each is its own
    # fix: repairing the group would leave the variant's own declaration broken.
    with pytest.raises(ConfigError) as info:
        validate_schema(UsesOverridden)
    assert len([issue for issue in info.value.issues if issue.path == "section.value"]) == 2


def test_a_field_the_group_declares_is_reported_once():
    # Every variant inherits the group's fields, so a walk per variant restates
    # what the group's own walk already reported.
    with pytest.raises(ConfigError) as info:
        validate_schema(UsesShared)
    assert len([issue for issue in info.value.issues if issue.path == "section.broken"]) == 1


# --- the guards that end a section before its keys are read ---------------------


@dataclass
class Guarded(ConfigChoice, tag_key="kind"):
    """Group reached by a mapping that ends before its keys are read."""


@dataclass
class GuardedOne(Guarded, tag="one"):
    """The one variant behind the guarded group, nesting the group again.

    One level per hop, so a chain of these puts its innermost section at a
    depth the test picks exactly.
    """

    value: int = 1
    inner: Guarded | None = None


@dataclass
class GuardedRoot(ConfigNode):
    """Root holding the guarded group."""

    section: Guarded


def test_a_mapping_that_reaches_itself_reports_the_cycle():
    # The cycle is what ends this section, and it is reported ahead of the
    # selection the mapping also lacks.
    data: dict[str, Any] = {}
    data["section"] = data
    assert _issues(GuardedRoot, data) == [("section", plain_cycle_message())]


def test_a_mapping_past_the_nesting_budget_reports_the_depth():
    # The innermost mapping omits its selection and also sits past the budget.
    # The budget is what ends the section, so the budget is what the report
    # names rather than the selection it also lacks.
    # One hop per level, so the innermost mapping lands at exactly the budget:
    # its parent is the last section within it, and the section it holds is the
    # first one past it.
    deep: Any = {}
    for _ in range(MAX_PLAIN_DEPTH - 1):
        deep = {"kind": "one", "inner": deep}
    reported = _issues(GuardedRoot, {"section": deep})
    assert reported[-1][1] == plain_depth_message()
    assert all("missing required value" not in message for _, message in reported)


# --- shared fields under a failed selection ------------------------------------


@dataclass
class Shared(ConfigChoice, tag_key="kind"):
    """Group declaring a field every variant inherits."""

    shared: int = 1


@dataclass
class SharedOnly(Shared, tag="only"):
    """The one variant the group stands for."""

    own: int = 2


@dataclass
class SharedRoot(ConfigNode):
    """Root holding a shared-field group beside a sibling of its own."""

    section: Shared
    sibling: int = 0


def test_a_shared_field_is_reported_beside_an_unknown_selection():
    # The group's own fields are inherited by every variant, so their values are
    # judged against the same annotation whichever variant the selection names.
    # One pass reports them alongside the selection that failed.
    assert _issues(SharedRoot, {"section": {"kind": "bogus", "shared": "x"}, "sibling": "y"}) == [
        ("section.kind", "expected one of 'only', got 'bogus'"),
        ("section.shared", "expected int, got str"),
        ("sibling", "expected int, got str"),
    ]


def test_a_shared_field_is_reported_beside_a_missing_selection():
    reported = _issues(SharedRoot, {"section": {"shared": "x"}})
    assert ("section.shared", "expected int, got str") in reported
    assert reported[0][0] == "section.kind"


def test_a_variant_field_waits_for_the_variant_that_declares_it():
    # Which further names the section carries is the selected variant's to say,
    # so a key outside the group's own fields is left to the load that names one.
    assert _issues(SharedRoot, {"section": {"kind": "bogus", "own": "x"}}) == [
        ("section.kind", "expected one of 'only', got 'bogus'"),
    ]


# --- a declaration the dataclass decorator rejected ----------------------------


@dataclass
class Corrected(ConfigChoice, tag_key="kind"):
    """Group a first declaration claims a string in and fails to complete."""

    common: int = 1


def test_a_string_returns_to_the_group_when_the_decorator_rejects_the_body():
    # Registration is recorded as the class is created, which is before the
    # decorator wrapping the declaration runs, so a decorator that raises leaves
    # a part-made class holding the string. The corrected declaration claims it.
    with pytest.raises(TypeError, match="follows default argument"):

        @dataclass
        class Half(Corrected, tag="one"):
            required: int  # pyrefly: ignore[bad-class-definition]

    @dataclass
    class Whole(Corrected, tag="one"):
        required: int = 2

    @dataclass
    class UsesCorrected(ConfigNode):
        section: Corrected

    built = from_dict(UsesCorrected, {"section": {"kind": "one", "required": 5}})
    assert isinstance(built.section, Whole)
    assert to_dict(built) == {"section": {"kind": "one", "common": 1, "required": 5}}


# --- one report per declaration ------------------------------------------------


@dataclass
class GroupCollision(ConfigChoice, tag_key="kind"):
    """Group declaring a field named for the key it selects under."""

    kind: str = "shadow"


@dataclass
class GroupCollisionOne(GroupCollision, tag="one"):
    """A variant inheriting the collision the group declares."""


@dataclass
class GroupCollisionTwo(GroupCollision, tag="two"):
    """A second variant inheriting the same collision."""


def test_a_collision_the_group_declares_is_reported_once():
    # One declaration to fix, so one report: the message names the class whose
    # body declares the field, which is what the variants' inherited copies
    # restate and the walk drops.
    @dataclass
    class UsesGroupCollision(ConfigNode):
        section: GroupCollision

    with pytest.raises(ConfigError) as info:
        validate_schema(UsesGroupCollision)
    collisions = [issue for issue in info.value.issues if "would name both" in issue.message]
    assert len(collisions) == 1
    assert "GroupCollision declares a field named 'kind'" in collisions[0].message


# --- dataclass flags across the hierarchy --------------------------------------


@dataclass(frozen=True)
class FrozenSchedule(ConfigChoice, tag_key="kind"):
    """Frozen group, which a frozen variant sits under."""

    total_steps: int = 10_000


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FrozenCosine(FrozenSchedule, tag="cosine"):
    """Frozen slotted variant carrying a weak-reference slot."""

    min_lr_ratio: float = 0.1


@dataclass(frozen=True)
class FrozenRoot(ConfigNode):
    """Frozen root holding the frozen group."""

    schedule: FrozenSchedule


def test_a_frozen_variant_sits_under_a_frozen_group():
    built = from_dict(FrozenRoot, {"schedule": {"kind": "cosine", "min_lr_ratio": 0.25}})
    assert isinstance(built.schedule, FrozenCosine)
    assert to_dict(built) == {"schedule": {"kind": "cosine", "total_steps": 10_000, "min_lr_ratio": 0.25}}
    assert from_dict(FrozenRoot, to_dict(built)) == built


def test_a_weakref_slot_variant_answers_a_weak_reference():
    built = from_dict(FrozenRoot, {"schedule": {"kind": "cosine"}})
    assert weakref.ref(built.schedule)() is built.schedule
    assert config_hash(built) != config_hash(from_dict(FrozenRoot, {"schedule": {"kind": "cosine", "total_steps": 1}}))


@dataclass
class Inheriting(ConfigChoice, tag_key="kind"):
    """Group whose initializer a variant of its own inherits."""

    value: int = 2


class InheritsInitializer(Inheriting, tag="same"):
    """Variant declaring no fields, so it builds through its group's initializer."""


def test_a_variant_that_inherits_its_group_initializer_keeps_its_string():
    # A variant declaring no fields of its own is complete, and its instances
    # carry the string their export writes, so the entry stays where marshal
    # finds it and a second class claiming the string is told so.
    live = InheritsInitializer()
    assert to_dict(live) == {"kind": "same", "value": 2}

    with pytest.raises(ConfigError, match="already carries"):

        @dataclass
        class Rival(Inheriting, tag="same"):
            value: int = 3

    assert to_dict(live) == {"kind": "same", "value": 2}
    assert from_dict(Inheriting, {"kind": "same", "value": 7}).value == 7


@dataclass
class Overriding(ConfigChoice, tag_key="kind"):
    """Group declaring a field one of its variants reads another way."""

    value: int = 1


@dataclass
class OverridingText(Overriding, tag="text"):
    """Variant redeclaring the group's field under its own annotation."""

    value: str = "x"  # pyrefly: ignore[bad-override-mutable-attribute]


@dataclass
class OverridingCount(Overriding, tag="count"):
    """Variant carrying the group's field as the group declares it."""


@dataclass
class OverridingRoot(ConfigNode):
    """Root holding the group whose field a variant redeclares."""

    section: Overriding


def test_a_field_a_variant_redeclares_waits_for_the_selection():
    # The selection settles which annotation the value is read against, so a
    # failed selection reports itself alone and the value stands as written.
    assert from_dict(OverridingRoot, {"section": {"kind": "text", "value": "hello"}}).section.value == "hello"
    assert _issues(OverridingRoot, {"section": {"kind": "bogus", "value": "hello"}}) == [
        ("section.kind", "expected one of 'count' | 'text', got 'bogus'"),
    ]


def test_a_selection_of_another_str_class_is_reported_on_both_routes():
    # A config file carries a plain str, so the class is what the selection is
    # matched on before the two strings are compared, whether the annotation
    # names the group or the variant.
    class Tag(str):
        pass

    assert _issues(Train, {"optimizer": {"algorithm": Tag("adamw")}}) == [
        ("optimizer.algorithm", "expected one of 'adamw' | 'sgd', got 'adamw'"),
    ]
    reported = _issues(AdamW, {"algorithm": Tag("adamw")})
    assert [path for path, _ in reported] == ["algorithm"]
    assert "the selection string AdamW carries" in reported[0][1]


@dataclass
class MixedFields:
    """Plain base a variant mixes in, carrying its own reading of a group name."""

    value: str = "text-default"


@dataclass
class Mixed(ConfigChoice, tag_key="kind"):
    """Group whose field name a variant takes another reading of from a base."""

    value: int = 1


@dataclass
class MixedText(MixedFields, Mixed, tag="text"):  # pyrefly: ignore[inconsistent-inheritance]
    """Variant reading the group's field name as its mixed-in base declares it."""


@dataclass
class MixedRoot(ConfigNode):
    """Root holding the group a variant mixes another reading into."""

    section: Mixed


def test_a_reading_a_variant_mixes_in_waits_for_the_selection():
    # The variant answers for the annotation the name carries on it, whether it
    # writes that annotation or takes it from a base, so the selection is what
    # settles which reading the value gets.
    assert from_dict(MixedRoot, {"section": {"kind": "text", "value": "hello"}}).section.value == "hello"
    assert _issues(MixedRoot, {"section": {"kind": "bogus", "value": "hello"}}) == [
        ("section.kind", "expected one of 'text', got 'bogus'"),
    ]
