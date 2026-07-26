"""Tests for the preflight guard on sets whose elements carry config sections.

Config dataclasses are unhashable, so a ``set`` or ``frozenset`` annotation naming
one -- directly, through a union, or inside an immutable ``tuple`` / ``frozenset``
shape -- is rejected at schema preflight with the collection remedy. Sets of
hashable values keep their deduplication, deterministic serialization, and round
trips.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
    from_dict,
    to_dict,
)


@dataclass
class Section:
    lr: float = 1e-3


@dataclass
class NodeSection(ConfigNode):
    lr: float = 1e-3


@dataclass
class BadSection:
    lr: complex = 0j


# --- rejected shapes ----------------------------------------------------------


@dataclass
class DirectFrozenset:
    sections: frozenset[Section] = field(default_factory=frozenset)


@dataclass
class DirectSet:
    sections: set[Section] = field(default_factory=set)


@dataclass
class NodeElementSet:
    sections: set[NodeSection] = field(default_factory=set)


@dataclass
class UnionElementSet:
    sections: set[Section | int] = field(default_factory=set)


@dataclass
class TupleShapeSet:
    sections: set[tuple[str, Section]] = field(default_factory=set)


@dataclass
class NestedFrozensetSet:
    sections: frozenset[frozenset[Section]] = field(default_factory=frozenset)


@dataclass
class MalformedSectionSet:
    sections: frozenset[BadSection] = field(default_factory=frozenset)


def _issues(config_cls: type[Any]) -> list[tuple[str, str]]:
    """Build ``config_cls`` from an empty mapping and return its reported issues.

    Args:
      config_cls (type[Any]): The schema class expected to fail preflight.

    Returns:
      list[tuple[str, str]]: One ``(path, message)`` pair per reported issue.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {})
    return [(issue.path, issue.message) for issue in info.value.issues]


def test_frozenset_of_sections_is_rejected_with_both_remedies():
    assert _issues(DirectFrozenset) == [
        (
            "sections",
            "config sections are unhashable, so frozenset[Section] cannot be built; use a list or tuple for "
            "the collection, and use config_hash(section) as the value-identity key when uniqueness matters",
        )
    ]


@pytest.mark.parametrize(
    ("config_cls", "rendered"),
    [
        (DirectSet, "set[Section]"),
        (NodeElementSet, "set[NodeSection]"),
        (UnionElementSet, "set[Section | int]"),
        (TupleShapeSet, "set[tuple[str, Section]]"),
        (NestedFrozensetSet, "frozenset[frozenset[Section]]"),
    ],
)
def test_section_bearing_sets_are_rejected_naming_the_written_annotation(config_cls: type[Any], rendered: str):
    reported = _issues(config_cls)
    assert reported[0][0] == "sections"
    assert f"so {rendered} cannot be built" in reported[0][1]


def test_nested_frozenset_reports_the_outer_and_inner_annotations():
    messages = [message for _path, message in _issues(NestedFrozensetSet)]
    assert any("so frozenset[frozenset[Section]] cannot be built" in message for message in messages)
    assert any("so frozenset[Section] cannot be built" in message for message in messages)


def test_a_defect_inside_the_section_still_aggregates():
    messages = [message for _path, message in _issues(MalformedSectionSet)]
    assert any("so frozenset[BadSection] cannot be built" in message for message in messages)
    assert any("unsupported field type complex" in message for message in messages)


# --- shapes that keep their current behavior ----------------------------------


@dataclass
class ScalarSets:
    tags: set[str] = field(default_factory=set)
    seeds: frozenset[int] = field(default_factory=frozenset)
    mixed: set[int | str] = field(default_factory=set)
    bare: set = field(default_factory=set)  # pyrefly: ignore[implicit-any-type-argument]
    opaque: set[Any] = field(default_factory=set)


def test_scalar_sets_build_and_deduplicate():
    built = from_dict(
        ScalarSets,
        {"tags": ["b", "a", "b"], "seeds": [2, 1, 2], "mixed": [1, "1"], "bare": ["x"], "opaque": [3]},
    )
    assert built.tags == {"a", "b"}
    assert built.seeds == frozenset({1, 2})
    assert built.mixed == {1, "1"}


def test_scalar_sets_serialize_deterministically_and_round_trip():
    built = from_dict(ScalarSets, {"tags": ["b", "a"], "seeds": [2, 1]})
    exported = to_dict(built)
    assert exported["tags"] == ["a", "b"]
    assert exported["seeds"] == [1, 2]
    assert from_dict(ScalarSets, exported) == built


# --- supported section collections --------------------------------------------


@dataclass
class SectionCollections:
    listed: list[Section] = field(default_factory=list)
    tupled: tuple[Section, ...] = ()


def test_lists_and_tuples_of_sections_remain_supported():
    built = from_dict(SectionCollections, {"listed": [{"lr": 0.1}], "tupled": [{"lr": 0.2}]})
    assert built.listed == [Section(lr=0.1)]
    assert built.tupled == (Section(lr=0.2),)
    assert from_dict(SectionCollections, to_dict(built)) == built
