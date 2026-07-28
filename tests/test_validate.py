"""Tests for the schema check that runs without any config data.

``validate_schema`` walks the tree a class declares, recursing into nested sections and
into sections held in containers, and reports every annotation outside the
supported set along with every authored default that does not already carry its
annotation's runtime type. No config data is read and no ``default_factory``
runs, so a schema can be checked before any file exists.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    assert_type,
)

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
)
from confingo.functional import (
    from_dict,
    validate_schema,
)


@dataclass
class Optimizer:
    name: str = "adamw"
    lr: float = 1e-3


@dataclass
class Training:
    optimizer: Optimizer
    epochs: int = 10


@dataclass
class Good:
    training: Training
    tags: list[str] = field(default_factory=list)


def test_a_supported_schema_validates_quietly():
    assert validate_schema(Good) is None


@dataclass
class BadLeaf:
    weird: complex = 0j


@dataclass
class InList:
    items: list[BadLeaf] = field(default_factory=list)


@dataclass
class InDict:
    items: dict[str, BadLeaf] = field(default_factory=dict)


@dataclass
class InTuple:
    items: tuple[BadLeaf, ...] = ()


@dataclass
class InSet:
    items: frozenset[int] = frozenset()
    section: BadLeaf | None = None


@dataclass
class Nested:
    outer: InList


@pytest.mark.parametrize(
    ("config_cls", "path"),
    [
        (InList, "items.weird"),
        (InDict, "items.weird"),
        (InTuple, "items.weird"),
        (InSet, "section.weird"),
        (Nested, "outer.items.weird"),
    ],
    ids=["list", "dict", "tuple", "union", "nested"],
)
def test_a_section_reached_through_a_container_is_inspected(config_cls: type[Any], path: str):
    """A container holding a dataclass still carries that dataclass's schema."""
    with pytest.raises(ConfigError) as info:
        validate_schema(config_cls)
    assert [issue.path for issue in info.value.issues] == [path]


@dataclass
class WrongDefault:
    count: int = "many"  # pyrefly: ignore[bad-assignment]


def test_an_authored_default_is_judged_without_any_data():
    with pytest.raises(ConfigError) as info:
        validate_schema(WrongDefault)
    assert "invalid authored default" in info.value.issues[0].message


FACTORY_RUNS: list[str] = []


def _counting_factory() -> int:
    """Record that the factory ran.

    Returns:
      int: A fixed value.
    """
    FACTORY_RUNS.append("ran")
    return 1


@dataclass
class HasFactory:
    count: int = field(default_factory=_counting_factory)


def test_validating_a_schema_runs_no_factory():
    FACTORY_RUNS.clear()
    validate_schema(HasFactory)
    assert FACTORY_RUNS == []


def test_the_context_names_the_schema():
    with pytest.raises(ConfigError) as info:
        validate_schema(WrongDefault)
    assert info.value.context == "config schema"
    with pytest.raises(ConfigError) as info:
        validate_schema(WrongDefault, context="sweep template")
    assert info.value.context == "sweep template"


def test_a_non_dataclass_entry_is_reported():
    with pytest.raises(ConfigError):
        validate_schema(int)


@dataclass
class GoodNode(ConfigNode):
    optimizer: Optimizer
    seed: int = 0


@dataclass
class BadNode(ConfigNode):
    weird: complex = 0j


def test_the_accessor_validates_the_class_it_is_reached_through():
    assert GoodNode.cfg.validate_schema() is None
    with pytest.raises(ConfigError):
        BadNode.cfg.validate_schema()


def test_the_accessor_carries_the_context_the_free_function_takes():
    with pytest.raises(ConfigError) as info:
        BadNode.cfg.validate_schema()
    assert info.value.context == "config schema"
    with pytest.raises(ConfigError) as info:
        BadNode.cfg.validate_schema(context="sweep template")
    assert info.value.context == "sweep template"


def test_the_accessor_validates_from_an_instance_too():
    built = GoodNode.cfg.from_dict({})
    assert built.cfg.validate_schema() is None


def test_a_nested_node_validates_its_own_subtree():
    @dataclass
    class Section(ConfigNode):
        lr: float = 1e-3

    assert validate_schema(Section) is None


def test_validation_reports_what_a_build_would_report():
    """The check a build runs before it builds is the check this exposes."""
    with pytest.raises(ConfigError) as from_validate:
        validate_schema(InList)
    with pytest.raises(ConfigError) as from_build:
        from_dict(InList, {})
    assert [(issue.path, issue.message) for issue in from_validate.value.issues] == [
        (issue.path, issue.message) for issue in from_build.value.issues
    ]


if TYPE_CHECKING:

    def _validate_typing_probe() -> None:
        """Pin what ``validate_schema`` answers, checked by the type checker alone.

        The wider public surface is pinned in ``test_typing_surface.py``; this
        body is never executed.
        """
        assert_type(validate_schema(Good), None)
        assert_type(validate_schema(Good, context="sweep template"), None)
        assert_type(GoodNode.cfg.validate_schema(), None)
        assert_type(GoodNode.cfg.validate_schema(context="sweep template"), None)
