"""Tests for what a failed union reports.

Selection is unchanged: declaration order, first member that coerces cleanly. When
no member fits, the report carries one branch's detail -- the branch whose trial
collected the fewest issues, declaration order breaking a tie -- under a summary
naming the whole union and the branch the detail came from.

A union names at most one config section, so the shapes here reach sections
through containers and beside scalars. Choosing among sections is what a variant
group answers, covered in ``test_choice_groups``.
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

from confingo import ConfigError
from confingo.functional import from_dict


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
    optimizer: list[AdamW] | list[SGD] = field(default_factory=list)
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
    built = from_dict(Root, {"optimizer": [{"kind": "adamw", "lr": 0.1}]})
    assert isinstance(built.optimizer[0], AdamW)
    assert built.optimizer[0].lr == 0.1


def test_a_later_member_is_selected_when_the_first_does_not_fit():
    built = from_dict(Root, {"optimizer": [{"kind": "sgd", "momentum": 0.5}]})
    assert isinstance(built.optimizer[0], SGD)
    assert built.optimizer[0].momentum == 0.5


def test_the_winning_member_is_built_once():
    POST_INIT_CALLS.clear()
    from_dict(Root, {"optimizer": [{"kind": "sgd"}]})
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
    assert _issues({"optimizer": [{"kind": "adamww", "lr": "fast"}]}) == [
        ("optimizer", "expected list[AdamW] | list[SGD]; best match list[AdamW] failed with 2 issues"),
        ("optimizer.0.kind", "expected one of 'adamw', got 'adamww'"),
        ("optimizer.0.lr", "expected float, got str"),
    ]


def test_the_branch_with_the_fewest_issues_supplies_the_detail():
    # AdamW fails three times here (discriminator, the momentum key it has no
    # field for, and lr); SGD fails on lr alone, so SGD supplies the detail
    # despite being declared second.
    reported = _issues({"optimizer": [{"kind": "sgd", "momentum": 0.5, "lr": "fast"}]})
    assert reported == [
        ("optimizer", "expected list[AdamW] | list[SGD]; best match list[SGD] failed with 1 issue"),
        ("optimizer.0.lr", "expected float, got str"),
    ]


def test_an_equal_count_tie_goes_to_the_first_declared_member():
    # Each member fails once on the discriminator typo alone.
    reported = _issues({"optimizer": [{"kind": "rmsprop"}]})
    assert reported[0] == (
        "optimizer",
        "expected list[AdamW] | list[SGD]; best match list[AdamW] failed with 1 issue",
    )
    assert reported[1] == ("optimizer.0.kind", "expected one of 'adamw', got 'rmsprop'")


def test_one_issue_is_singular_and_several_are_plural():
    singular = _issues({"optimizer": [{"kind": "rmsprop"}]})[0][1]
    plural = _issues({"optimizer": [{"kind": "rmsprop", "lr": "fast"}]})[0][1]
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
    optimizer: list[AdamW] | list[SGD] | None = None


def test_an_optional_multi_member_union_accepts_none_and_still_reports_a_branch():
    assert from_dict(OptionalUnion, {"optimizer": None}).optimizer is None
    reported = _issues({"optimizer": [{"kind": "rmsprop"}]}, OptionalUnion)
    assert reported[0] == (
        "optimizer",
        "expected list[AdamW] | list[SGD] | None; best match list[AdamW] failed with 1 issue",
    )
    assert reported[1] == ("optimizer.0.kind", "expected one of 'adamw', got 'rmsprop'")


# --- a section beside a scalar ------------------------------------------------


@dataclass
class Mixed:
    optimizer: AdamW | int = 0


def test_a_section_beside_a_scalar_selects_by_the_form_the_file_carried():
    assert from_dict(Mixed, {"optimizer": 4}).optimizer == 4
    assert from_dict(Mixed, {"optimizer": {"kind": "adamw", "lr": 0.5}}).optimizer == AdamW(lr=0.5)


# --- paths and aggregation ----------------------------------------------------


@dataclass
class Listed:
    items: list[AdamW | int] = field(default_factory=list)
    seed: int = 0


def test_a_failed_union_inside_a_list_carries_the_element_index():
    reported = _issues({"items": [{"kind": "adamw"}, {"kind": "nope"}], "seed": "bad"}, Listed)
    assert reported[0] == ("items.1", "expected AdamW | int; best match AdamW failed with 1 issue")
    assert reported[1] == ("items.1.kind", "expected one of 'adamw', got 'nope'")
    assert reported[2] == ("seed", "expected int, got str")
