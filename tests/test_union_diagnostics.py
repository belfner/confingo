"""Tests for what a failed union reports.

Selection is unchanged: declaration order, first member that coerces cleanly. When
no member fits, the report carries one branch's detail -- the branch whose trial
collected the fewest issues, declaration order breaking a tie -- under a summary
naming the whole union and the branch the detail came from.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Literal,
)

import pytest

from confingo import (
    ConfigError,
    from_dict,
)


POST_INIT_CALLS: list[str] = []


@dataclass
class AdamW:
    kind: Literal["adamw"] = "adamw"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        POST_INIT_CALLS.append("AdamW")


@dataclass
class SGD:
    kind: Literal["sgd"] = "sgd"
    lr: float = 1e-3
    momentum: float = 0.9

    def __post_init__(self) -> None:
        POST_INIT_CALLS.append("SGD")


@dataclass
class Root:
    optimizer: AdamW | SGD = field(default_factory=AdamW)
    seed: int = 0


def _issues(data: dict[str, Any], config_cls: type[Any] = Root) -> list[tuple[str, str]]:
    """Build ``config_cls`` from ``data`` and return its issues in report order.

    Args:
      data (dict[str, Any]): The mapping to build from.
      config_cls (type[Any] = Root): The schema class expected to fail.

    Returns:
      list[tuple[str, str]]: One ``(path, message)`` pair per issue, in order.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, data)
    return [(issue.path, issue.message) for issue in info.value.issues]


# --- selection is unchanged ---------------------------------------------------


def test_the_first_declared_member_that_fits_wins():
    built = from_dict(Root, {"optimizer": {"kind": "adamw", "lr": 0.1}})
    assert isinstance(built.optimizer, AdamW)
    assert built.optimizer.lr == 0.1


def test_a_later_member_is_selected_when_the_first_does_not_fit():
    built = from_dict(Root, {"optimizer": {"kind": "sgd", "momentum": 0.5}})
    assert isinstance(built.optimizer, SGD)
    assert built.optimizer.momentum == 0.5


def test_the_winning_member_is_built_once():
    POST_INIT_CALLS.clear()
    from_dict(Root, {"optimizer": {"kind": "sgd"}})
    # AdamW's trial fails on the discriminator before its constructor is reached,
    # and SGD's trial value is kept rather than rebuilt, so the selected member is
    # constructed exactly once and the rejected one not at all.
    assert POST_INIT_CALLS == ["SGD"]


def test_an_optional_single_member_union_reports_its_own_detail():
    @dataclass
    class Optional_:
        value: int | None = None

    assert _issues({"value": "x"}, Optional_) == [("value", "expected int, got str")]


# --- failure reports one branch's detail --------------------------------------


def test_the_summary_names_the_union_and_the_branch_the_detail_came_from():
    assert _issues({"optimizer": {"kind": "adamww", "lr": "fast"}}) == [
        ("optimizer", "expected AdamW | SGD; best match AdamW failed with 2 issues"),
        ("optimizer.kind", "expected one of 'adamw', got 'adamww'"),
        ("optimizer.lr", "expected float, got str"),
    ]


def test_the_branch_with_the_fewest_issues_supplies_the_detail():
    # AdamW fails three times here (discriminator, the momentum key it has no
    # field for, and lr); SGD fails on lr alone, so SGD supplies the detail
    # despite being declared second.
    reported = _issues({"optimizer": {"kind": "sgd", "momentum": 0.5, "lr": "fast"}})
    assert reported == [
        ("optimizer", "expected AdamW | SGD; best match SGD failed with 1 issue"),
        ("optimizer.lr", "expected float, got str"),
    ]


def test_an_equal_count_tie_goes_to_the_first_declared_member():
    # Each member fails once on the discriminator typo alone.
    reported = _issues({"optimizer": {"kind": "rmsprop"}})
    assert reported[0] == ("optimizer", "expected AdamW | SGD; best match AdamW failed with 1 issue")
    assert reported[1] == ("optimizer.kind", "expected one of 'adamw', got 'rmsprop'")


def test_one_issue_is_singular_and_several_are_plural():
    singular = _issues({"optimizer": {"kind": "rmsprop"}})[0][1]
    plural = _issues({"optimizer": {"kind": "rmsprop", "lr": "fast"}})[0][1]
    assert singular.endswith("failed with 1 issue")
    assert plural.endswith("failed with 2 issues")


def test_a_scalar_union_keeps_its_type_summary():
    @dataclass
    class Scalars:
        value: int | str = 0

    assert _issues({"value": [1]}, Scalars) == [
        ("value", "expected int | str; best match int failed with 1 issue"),
        ("value", "expected int, got list"),
    ]


@dataclass
class OptionalUnion:
    optimizer: AdamW | SGD | None = None


def test_an_optional_multi_member_union_accepts_none_and_still_reports_a_branch():
    assert from_dict(OptionalUnion, {"optimizer": None}).optimizer is None
    reported = _issues({"optimizer": {"kind": "rmsprop"}}, OptionalUnion)
    assert reported[0] == ("optimizer", "expected AdamW | SGD | None; best match AdamW failed with 1 issue")
    assert reported[1] == ("optimizer.kind", "expected one of 'adamw', got 'rmsprop'")


# --- paths and aggregation ----------------------------------------------------


@dataclass
class Listed:
    items: list[AdamW | SGD] = field(default_factory=list)
    seed: int = 0


def test_a_failed_union_inside_a_list_carries_the_element_index():
    reported = _issues({"items": [{"kind": "adamw"}, {"kind": "nope"}], "seed": "bad"}, Listed)
    assert reported[0] == ("items.1", "expected AdamW | SGD; best match AdamW failed with 1 issue")
    assert reported[1] == ("items.1.kind", "expected one of 'adamw', got 'nope'")
    assert reported[2] == ("seed", "expected int, got str")


# --- structurally identical variants ------------------------------------------


@dataclass
class WarmupCosine:
    schedule: Literal["cosine"] = "cosine"
    warmup_steps: int = 500
    total_steps: int = 10_000


@dataclass
class WarmupLinear:
    schedule: Literal["linear"] = "linear"
    warmup_steps: int = 500
    total_steps: int = 10_000


@dataclass
class Scheduled:
    schedule: WarmupCosine | WarmupLinear = field(default_factory=WarmupCosine)


def test_identical_variants_differing_only_by_discriminator_report_through_the_first():
    # Both members carry the same fields, so a typo in the discriminator fails
    # each exactly once and nothing distinguishes them by issue count. The
    # summary still names the whole union, so the reader sees both options.
    reported = _issues({"schedule": {"schedule": "cosinus"}}, Scheduled)
    assert reported == [
        ("schedule", "expected WarmupCosine | WarmupLinear; best match WarmupCosine failed with 1 issue"),
        ("schedule.schedule", "expected one of 'cosine', got 'cosinus'"),
    ]


def test_a_valid_second_identical_variant_still_selects_cleanly():
    built = from_dict(Scheduled, {"schedule": {"schedule": "linear", "warmup_steps": 100}})
    assert isinstance(built.schedule, WarmupLinear)
    assert built.schedule.warmup_steps == 100
